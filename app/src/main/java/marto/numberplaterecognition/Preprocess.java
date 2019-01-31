package marto.numberplaterecognition;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

class Preprocess {
    private static final Size GAUSSIAN_SMOOTH_FILTER_SIZE = new Size(5, 5);
    private static final int ADAPTIVE_THRESH_BLOCK_SIZE = 19;
    private static final int ADAPTIVE_THRESH_WEIGHT = 9;

    static void preprocess(Mat imgOriginal, Mat imgGrayscale, Mat imgThresh) {
        imgGrayscale = extractValue(imgOriginal);

        Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);
        Mat imgBlurred = new Mat();
        Imgproc.GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);

        Imgproc.adaptiveThreshold(imgBlurred, imgThresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);
    }

    private static Mat extractValue(Mat imgOriginal) {
        Mat imgHSV = new Mat();
        List<Mat> vectorOfHSVImages = new ArrayList<>();
        Mat imgValue;

        Imgproc.cvtColor(imgOriginal, imgHSV, Imgproc.COLOR_BGR2HSV);
        Core.split(imgHSV, vectorOfHSVImages);
        imgValue = vectorOfHSVImages.get(2);

        return imgValue;
    }

    private static Mat maximizeContrast(Mat imgGrayscale) {
        Mat imgTopHat = new Mat();
        Mat imgBlackHat = new Mat();
        Mat imgGrayscalePlusTopHat = new Mat();
        Mat imgGrayscalePlusTopHatMinusBlackHat = new Mat();

        Mat structuringElement = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(3, 3));

        Imgproc.morphologyEx(imgGrayscale, imgTopHat, Imgproc.MORPH_TOPHAT, structuringElement);
        Imgproc.morphologyEx(imgGrayscale, imgBlackHat, Imgproc.MORPH_BLACKHAT, structuringElement);

        Core.add(imgGrayscale, imgTopHat, imgGrayscalePlusTopHat);
        Core.subtract(imgGrayscalePlusTopHat, imgBlackHat, imgGrayscalePlusTopHatMinusBlackHat);

        return imgGrayscalePlusTopHatMinusBlackHat;
    }
}
