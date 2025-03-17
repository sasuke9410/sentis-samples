using System;
using Unity.Mathematics;
using Unity.Sentis;
using UnityEngine;

public class FaceDetectionToPotato : MonoBehaviour
{
    public FacePreview[] facePreviews;
    public ImagePreview imagePreview;
    public Texture2D imageTexture;
    public ModelAsset faceDetector;
    public TextAsset anchorsCSV;
    public Texture2D potatoTexture; // ジャガイモの画像を保持するフィールドを追加

    public float iouThreshold = 0.3f;
    public float scoreThreshold = 0.5f;

    const int k_NumAnchors = 896;
    float[,] m_Anchors;

    const int k_NumKeypoints = 6;
    const int detectorInputSize = 128;

    Worker m_FaceDetectorWorker;
    Tensor<float> m_DetectorInput;
    Awaitable m_DetectAwaitable;

    float m_TextureWidth;
    float m_TextureHeight;

    public async void Start()
    {
        m_Anchors = BlazeUtils.LoadAnchors(anchorsCSV.text, k_NumAnchors);

        var faceDetectorModel = ModelLoader.Load(faceDetector);

        // post process the model to filter scores + nms select the best faces
        var graph = new FunctionalGraph();
        var input = graph.AddInput(faceDetectorModel, 0);
        var outputs = Functional.Forward(faceDetectorModel, 2 * input - 1);
        var boxes = outputs[0]; // (1, 896, 16)
        var scores = outputs[1]; // (1, 896, 1)
        var anchorsData = new float[k_NumAnchors * 4];
        Buffer.BlockCopy(m_Anchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
        var anchors = Functional.Constant(new TensorShape(k_NumAnchors, 4), anchorsData);
        var idx_scores_boxes = BlazeUtils.NMSFiltering(boxes, scores, anchors, detectorInputSize, iouThreshold, scoreThreshold);
        faceDetectorModel = graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);

        m_FaceDetectorWorker = new Worker(faceDetectorModel, BackendType.GPUCompute);

        m_DetectorInput = new Tensor<float>(new TensorShape(1, detectorInputSize, detectorInputSize, 3));

        while (true)
        {
            try
            {
                m_DetectAwaitable = Detect(imageTexture);
                await m_DetectAwaitable;
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        m_FaceDetectorWorker.Dispose();
        m_DetectorInput.Dispose();
    }

    Vector3 ImageToWorld(Vector2 position)
    {
        return (position - 0.5f * new Vector2(m_TextureWidth, m_TextureHeight)) / m_TextureHeight;
    }

    async Awaitable Detect(Texture texture)
    {
        m_TextureWidth = texture.width;
        m_TextureHeight = texture.height;
        imagePreview.SetTexture(texture);

        var size = Mathf.Max(texture.width, texture.height);

        // The affine transformation matrix to go from tensor coordinates to image coordinates
        var scale = size / (float)detectorInputSize;
        var M = BlazeUtils.mul(BlazeUtils.TranslationMatrix(0.5f * (new Vector2(texture.width, texture.height) + new Vector2(-size, size))), BlazeUtils.ScaleMatrix(new Vector2(scale, -scale)));
        BlazeUtils.SampleImageAffine(texture, m_DetectorInput, M);

        m_FaceDetectorWorker.Schedule(m_DetectorInput);

        var outputIndicesAwaitable = (m_FaceDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
        var outputScoresAwaitable = (m_FaceDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
        var outputBoxesAwaitable = (m_FaceDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

        using var outputIndices = await outputIndicesAwaitable;
        using var outputScores = await outputScoresAwaitable;
        using var outputBoxes = await outputBoxesAwaitable;

        var numFaces = outputIndices.shape.length;

        for (var i = 0; i < facePreviews.Length; i++)
        {
            var active = i < numFaces;
            facePreviews[i].SetActive(active);
            if (!active)
                continue;

            var idx = outputIndices[i];

            var anchorPosition = detectorInputSize * new float2(m_Anchors[idx, 0], m_Anchors[idx, 1]);

            var box_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBoxes[0, i, 0], outputBoxes[0, i, 1]));
            var boxTopRight_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBoxes[0, i, 0] + 0.5f * outputBoxes[0, i, 2], outputBoxes[0, i, 1] + 0.5f * outputBoxes[0, i, 3]));

            var boxSize = 2f * (boxTopRight_ImageSpace - box_ImageSpace);
            facePreviews[i].SetBoundingBox(true, ImageToWorld(box_ImageSpace), boxSize / texture.height);

            for (var j = 0; j < k_NumKeypoints; j++)
            {
                var position_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBoxes[0, i, 4 + 2 * j + 0], outputBoxes[0, i, 4 + 2 * j + 1]));
                facePreviews[i].SetKeypoint(j, true, ImageToWorld(position_ImageSpace));
            }

            // ジャガイモの画像を表示
            var potatoPosition = ImageToWorld(box_ImageSpace);
            var potatoSize = boxSize / texture.height;
            DisplayPotato(potatoPosition, potatoSize);
        }

        // if no faces are recognized then the awaitable outputs return synchronously so we need to add an extra frame await here to allow the main thread to run
        if (numFaces == 0)
            await Awaitable.NextFrameAsync();
    }

    void DisplayPotato(Vector3 position, Vector2 size)
    {
        // 既存のジャガイモオブジェクトがある場合は削除
    var existingPotato = GameObject.Find("Potato");
    if (existingPotato != null)
    {
        Destroy(existingPotato);
    }

    // ジャガイモの画像を表示するためのロジックを追加
    var potatoObject = new GameObject("Potato");
    var renderer = potatoObject.AddComponent<SpriteRenderer>();
    renderer.sprite = Sprite.Create(potatoTexture, new Rect(0, 0, potatoTexture.width, potatoTexture.height), new Vector2(0.5f, 0.5f));
    potatoObject.transform.position = position;
    potatoObject.transform.localScale = new Vector3(size.x, size.y, 1);
    }

    void OnDestroy()
    {
        m_DetectAwaitable.Cancel();
    }
}
