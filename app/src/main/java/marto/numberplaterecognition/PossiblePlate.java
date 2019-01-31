package marto.numberplaterecognition;

import org.opencv.core.Mat;
import org.opencv.core.RotatedRect;

public class PossiblePlate implements Comparable<PossiblePlate> {
    private Mat imgPlate;
    private Mat imgGrayscale;
    private Mat imgThresh;
    private RotatedRect rrLocationOfPlateInScene;
    private String strChars;

    public  PossiblePlate() {
        imgPlate = new Mat();
        imgGrayscale = new Mat();
        imgThresh = new Mat();
        rrLocationOfPlateInScene = new RotatedRect();
        strChars = "";
    }

    public Mat getImgPlate() {
        return imgPlate;
    }

    public void setImgPlate(Mat imgPlate) {
        this.imgPlate = imgPlate;
    }

    public Mat getImgGrayscale() {
        return imgGrayscale;
    }

    public void setImgGrayscale(Mat imgGrayscale) {
        this.imgGrayscale = imgGrayscale;
    }

    public Mat getImgThresh() {
        return imgThresh;
    }

    public void setImgThresh(Mat imgThresh) {
        this.imgThresh = imgThresh;
    }

    public RotatedRect getRrLocationOfPlateInScene() {
        return rrLocationOfPlateInScene;
    }

    public void setRrLocationOfPlateInScene(RotatedRect rrLocationOfPlateInScene) {
        this.rrLocationOfPlateInScene = rrLocationOfPlateInScene;
    }

    public String getStrChars() {
        return strChars;
    }

    public String getStrCharsReverse() {
        return new StringBuilder(strChars).reverse().toString();
    }

    public void setStrChars(String strChars) {
        this.strChars = strChars;
    }

    @Override
    public int compareTo(PossiblePlate possiblePlate) {
        return this.getStrChars().length() - possiblePlate.getStrChars().length();
    }
}
