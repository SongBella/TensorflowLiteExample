package com.example.tensorliteex.draw;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.divyanshu.draw.widget.DrawView;
import com.example.tensorliteex.R;
import com.example.tensorliteex.tflite.*;

import java.io.IOException;
import java.util.Locale;

public class DrawActivity extends AppCompatActivity {

    Classifier cls; // Classifier 객체를 선언합니다.

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_draw);

        DrawView drawView = findViewById(R.id.drawView); // 그림을 그릴 수 있는 뷰를 찾아서 초기화합니다.
        drawView.setStrokeWidth(100.0f); // 그림을 그릴 때 사용되는 선의 굵기를 설정합니다.
        drawView.setBackgroundColor(Color.BLACK); // 그림판의 배경색을 검정색으로 설정합니다.
        drawView.setColor(Color.WHITE); // 그려지는 선의 색상을 흰색으로 설정합니다.

        TextView resultView = findViewById(R.id.resultView); // 결과를 표시할 TextView를 찾아 초기화합니다.

        Button classifyBtn = findViewById(R.id.classifyBtn);
        classifyBtn.setOnClickListener(v -> {
            Bitmap image = drawView.getBitmap(); // 그림을 Bitmap 형식으로 얻어옵니다.

            Pair<Integer, Float> res = cls.classify(image); // Classifier를 사용하여 그림을 분류합니다.
            String outStr = String.format(Locale.ENGLISH, "%d, %.0f%%", res.first, res.second * 100.0f); // 결과를 문자열로 변환합니다.
            resultView.setText(outStr); // 결과를 화면에 표시합니다.
        });

        Button clearBtn = findViewById(R.id.clearBtn);
        clearBtn.setOnClickListener(v -> {
            drawView.clearCanvas(); // 그림을 지우는 버튼을 클릭하면 그림을 지웁니다.
        });

        cls = new Classifier(this); // Classifier 객체를 생성합니다.
        try {
            cls.init(); // Classifier를 초기화합니다.
        } catch(IOException ioe) {
            Log.d("DigitClassifier", "failed to init Classifier", ioe); // 초기화 중 오류가 발생하면 로그에 오류 메시지를 출력합니다.
        }

    }

    @Override
    protected void onDestroy() {
        cls.finish(); // 액티비티가 파괴될 때 Classifier를 종료합니다.
        super.onDestroy();
    }
}