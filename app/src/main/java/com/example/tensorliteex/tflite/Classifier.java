package com.example.tensorliteex.tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Pair;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class Classifier {
    private static final String MODEL_NAME = "keras_model_cnn.tflite"; // 사용할 TensorFlow Lite 모델의 파일 이름

    Context context; // Android 애플리케이션 컨텍스트
    Interpreter interpreter = null; // TensorFlow Lite 모델을 실행하기 위한 인터프리터
    int modelInputWidth, modelInputHeight, modelInputChannel; // 모델의 입력 이미지 크기 및 채널 수
    int modelOutputClasses; // 모델의 출력 클래스 수

    public Classifier(Context context) {
        this.context = context;
    }

    public void init() throws IOException {
        ByteBuffer model = loadModelFile(MODEL_NAME); // 모델 파일을 로드하여 ByteBuffer로 읽어옵니다.
        model.order(ByteOrder.nativeOrder());
        interpreter = new Interpreter(model); // 모델을 인터프리터로 초기화합니다.

        initModelShape(); // 모델의 입력 및 출력 모양(크기)를 초기화합니다.
    }

    private ByteBuffer loadModelFile(String modelName) throws IOException {
        // 모델 파일을 읽어오는 함수
        AssetManager am = context.getAssets();
        AssetFileDescriptor afd = am.openFd(modelName);
        FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
        FileChannel fc = fis.getChannel();
        long startOffset = afd.getStartOffset();
        long declaredLength = afd.getDeclaredLength();

        return fc.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void initModelShape() {
        // 모델의 입력 및 출력 텐서의 모양(크기)을 초기화합니다.
        Tensor inputTensor = interpreter.getInputTensor(0);
        int[] inputShape = inputTensor.shape();
        modelInputChannel = inputShape[0];
        modelInputWidth = inputShape[1];
        modelInputHeight = inputShape[2];

        Tensor outputTensor = interpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        modelOutputClasses = outputShape[1];
    }

    private Bitmap resizeBitmap(Bitmap bitmap) {
        // 입력 이미지 크기로 비트맵 이미지를 조절합니다.
        return Bitmap.createScaledBitmap(bitmap, modelInputWidth, modelInputHeight, false);
    }

    private ByteBuffer convertBitmapToGrayByteBuffer(Bitmap bitmap) {
        // 비트맵 이미지를 흑백 이미지로 변환하고 ByteBuffer로 변환합니다.
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bitmap.getByteCount());
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[bitmap.getWidth() * bitmap.getHeight()];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int pixel : pixels) {
            int r = pixel >> 16 & 0xFF;
            int g = pixel >> 8 & 0xFF;
            int b = pixel & 0xFF;

            float avgPixelValue = (r + g + b) / 3.0f;
            float normalizedPixelValue = avgPixelValue / 255.0f;

            byteBuffer.putFloat(normalizedPixelValue);
        }

        return byteBuffer;
    }

    public Pair<Integer, Float> classify(Bitmap image) {
        // 이미지를 분류하고 결과를 반환합니다.
        ByteBuffer buffer = convertBitmapToGrayByteBuffer(resizeBitmap(image));

        float[][] result = new float[1][modelOutputClasses];

        interpreter.run(buffer, result);

        return argmax(result[0]);
    }

    private Pair<Integer, Float> argmax(float[] array) {
        // 배열에서 최댓값과 해당 인덱스를 찾아 반환합니다.
        int argmax = 0;
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            float f = array[i];
            if (f > max) {
                argmax = i;
                max = f;
            }
        }
        return new Pair<>(argmax, max);
    }

    public void finish() {
        // Classifier 사용이 끝나면 인터프리터를 닫습니다.
        if (interpreter != null)
            interpreter.close();
    }
}
