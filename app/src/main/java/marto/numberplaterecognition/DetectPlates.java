package marto.numberplaterecognition;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class DetectPlates {

    private static final double PLATE_WIDTH_PADDING_FACTOR = 1.3;
    private static final double PLATE_HEIGHT_PADDING_FACTOR = 1.5;

    private DetectChars detectChars;

    DetectPlates() {
        detectChars = new DetectChars();
    }

    List<PossiblePlate> detectPlatesInScene(Mat imgOriginalScene) {
        List<PossiblePlate> listOfPossiblePlates = new ArrayList<>();

        Mat imgGrayscaleScene = new Mat();
        Mat imgThreshScene = new Mat();

        Preprocess.preprocess(imgOriginalScene, imgGrayscaleScene, imgThreshScene);

        List<PossibleChar> listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene);
        List<List<PossibleChar>> listOfMatchingCharsInScene = detectChars.findMatchingChars(listOfPossibleCharsInScene);

        // for each group of matching chars attempt to extract plate
        for (List<PossibleChar> listOfMatchingChars : listOfMatchingCharsInScene) {
            PossiblePlate possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars);

            if (!possiblePlate.getImgPlate().empty())
                listOfPossiblePlates.add(possiblePlate);
        }

        return listOfPossiblePlates;
    }

    private List<PossibleChar> findPossibleCharsInScene(Mat imgThresh) {
        List<PossibleChar> listOfPossibleChars = new ArrayList<>();

        Mat imgThreshCopy = imgThresh.clone();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(imgThreshCopy, contours, imgThreshCopy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        for (int i = 0; i < contours.size(); i++) {
            PossibleChar possibleChar = new PossibleChar(contours.get(i));

            if (detectChars.checkIfPossibleChar(possibleChar))
                listOfPossibleChars.add(possibleChar);
        }
        return listOfPossibleChars;
    }

    private PossiblePlate extractPlate(Mat imgOriginal, List<PossibleChar> listOfMatchingChars) {
        PossiblePlate possiblePlate = new PossiblePlate();

        Collections.sort(listOfMatchingChars);

        // Center point of the plate
        double dblPlateCenterX = (listOfMatchingChars.get(0).getIntCenterX() + listOfMatchingChars.get(listOfMatchingChars.size() - 1).getIntCenterX()) / 2.0;
        double dblPlateCenterY = (listOfMatchingChars.get(0).getIntCenterY() + listOfMatchingChars.get(listOfMatchingChars.size() - 1).getIntCenterY()) / 2.0;
        Point p2dPlateCenter = new Point(dblPlateCenterX, dblPlateCenterY);

        // Plate width and height
        int intPlateWidth = (int) Math.abs((listOfMatchingChars.get(listOfMatchingChars.size() - 1).getBoundingRect().x +
                listOfMatchingChars.get(listOfMatchingChars.size() - 1).getBoundingRect().width -
                listOfMatchingChars.get(0).getBoundingRect().x) * PLATE_WIDTH_PADDING_FACTOR);

        double intTotalOfCharHeights = 0;
        for (PossibleChar matchingChar : listOfMatchingChars)
            intTotalOfCharHeights += matchingChar.getBoundingRect().height;

        double dblAverageCharHeight = intTotalOfCharHeights / listOfMatchingChars.size();

        int intPlateHeight = (int) Math.abs(dblAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR);

        // calculate correction angle of plate region
        double dblOpposite = listOfMatchingChars.get(listOfMatchingChars.size() - 1).getIntCenterY() - listOfMatchingChars.get(0).getIntCenterY();
        double dblHypotenuse = new DetectChars().distanceBetweenChars(listOfMatchingChars.get(0), listOfMatchingChars.get(listOfMatchingChars.size() - 1));
        double dblCorrectionAngleInRad = Math.asin(dblOpposite / dblHypotenuse);
        double dblCorrectionAngleInDeg = dblCorrectionAngleInRad * (180.0 / Math.PI);

        // assign rotated rect member variable of possible plate
        possiblePlate.setRrLocationOfPlateInScene(new RotatedRect(p2dPlateCenter, new Size((float) intPlateWidth, (float) intPlateHeight), dblCorrectionAngleInDeg));

        Mat rotationMatrix;
        Mat imgRotated = new Mat();
        Mat imgCropped = new Mat();

        rotationMatrix = Imgproc.getRotationMatrix2D(p2dPlateCenter, dblCorrectionAngleInDeg, 1.0);

        // rotate the entire image
        Imgproc.warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());

        // crop out the actual plate portion of the rotated image
        Imgproc.getRectSubPix(imgRotated, possiblePlate.getRrLocationOfPlateInScene().size, possiblePlate.getRrLocationOfPlateInScene().center, imgCropped);

        possiblePlate.setImgPlate(imgCropped);
        return possiblePlate;
    }

}
