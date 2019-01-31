package marto.numberplaterecognition;

import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.DTrees;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.ml.NormalBayesClassifier;
import org.opencv.ml.SVM;
import org.opencv.ml.TrainData;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private TextView plateText;
    private ImageView imageView;
    private ProgressDialog pd;

    public static KNearest kNearest;

    public static final int PICK_IMAGE = 1;

    private boolean openCV = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        plateText = findViewById(R.id.plateText);
        imageView = findViewById(R.id.imageView);
        Button selectButton = findViewById(R.id.selectButton);

        selectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select image"), PICK_IMAGE);
            }
        });
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                // OpenCV loaded successfully
                case LoaderCallbackInterface.SUCCESS: {
                    openCV = true;
                    // KNN training
                    loadKNNDataAndTrainKNN();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };


    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == PICK_IMAGE && data.getData() != null) {
            try {
                InputStream image = getContentResolver().openInputStream(data.getData());
                if (openCV)
                    detectInImage(image);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    public void detectInImage(InputStream image) {

        BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
        bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;

        Bitmap bmp = BitmapFactory.decodeStream(image, null, bmpFactoryOptions);
        Mat src = new Mat();
        Utils.bitmapToMat(bmp, src);

        // Show Image for detect
        imageView.setImageBitmap(bmp);

        Mat originalImg = new Mat();
        Imgproc.cvtColor(src, originalImg, Imgproc.COLOR_BGRA2BGR);

        // If image is big resize for faster computing
        if (originalImg.size().height > 1024 || originalImg.size().width > 1024) {
            double scale = 1024 / originalImg.size().height ;
            Imgproc.resize(originalImg, originalImg, new Size(), scale, scale);
        }

        DetectPlates detectPlates = new DetectPlates();
        DetectChars detectChars = new DetectChars();

        // detect plates
        List<PossiblePlate> possiblePlates = detectPlates.detectPlatesInScene(originalImg);
        // detect chars in plates
        possiblePlates = detectChars.detectCharsInPlates(possiblePlates);

        if (possiblePlates.isEmpty()) {
            plateText.setText("No number plates detected");
        } else {
            // Sort possible plates in DESCENDING order (plate with most chars first)
            Collections.sort(possiblePlates);
            PossiblePlate licPlate = possiblePlates.get(0);

            if (licPlate.getStrChars().isEmpty())
                plateText.setText("No number plate found");
            else
                plateText.setText(licPlate.getStrCharsReverse());
        }
    }

    public void loadKNNDataAndTrainKNN() {
        kNearest = KNearest.create();

        // read in training classifications
        Mat matClassificationInts = new Mat(180, 1, CvType.CV_32F);
        int k = 0;
        try {
            InputStream inputStream = getAssets().open("classifications.txt");
            InputStreamReader inputStreamReader = new InputStreamReader(inputStream, "UTF-8");
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            String line = bufferedReader.readLine();
            while (line != null) {
                // Read 180 rows in 1 column
                matClassificationInts.put(k++, 0, Integer.valueOf(line));
                line = bufferedReader.readLine();
            }
        } catch (IOException ignored) {
        }

        // read in training images file
        List<Float> values = null;
        try {
            InputStream inputStream = getAssets().open("images.txt");
            InputStreamReader inputStreamReader = new InputStreamReader(inputStream, "UTF-8");
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            values = new ArrayList<>();
            String line = bufferedReader.readLine();
            while (line != null) {
                for (String s : line.split(" "))
                    values.add(Float.valueOf(s));
                line = bufferedReader.readLine();
            }
        } catch (IOException ignored) {
        }

        // read multiple images into single image
        Mat matTrainingImagesAsFlattenedFloats = new Mat(180, 600, CvType.CV_32F);
        k = 0;
        for (int i = 0; i < 180; ++i)
            for (int j = 0; j < 600; ++j)
                matTrainingImagesAsFlattenedFloats.put(i, j, values.get(k++));

        kNearest.setDefaultK(1);
        kNearest.train(matTrainingImagesAsFlattenedFloats, Ml.ROW_SAMPLE, matClassificationInts);
    }

}
