package marto.numberplaterecognition;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class DetectChars {
    // constants for checkIfPossibleChar
    private static final int MIN_PIXEL_WIDTH = 2;
    private static final int MIN_PIXEL_HEIGHT = 8;
    private static final double MIN_ASPECT_RATIO = 0.25;
    private static final double MAX_ASPECT_RATIO = 1.0;
    private static final int MIN_PIXEL_AREA = 80;

    // constants for comparing two chars
    private static final double MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0;
    private static final double MAX_CHANGE_IN_AREA = 0.5;
    private static final double MAX_CHANGE_IN_WIDTH = 0.8;
    private static final double MAX_CHANGE_IN_HEIGHT = 0.2;
    private static final double MAX_ANGLE_BETWEEN_CHARS = 12.0;

    private static final int MIN_NUMBER_OF_MATCHING_CHARS = 3;
    private static final int RESIZED_CHAR_IMAGE_WIDTH = 20;
    private static final int RESIZED_CHAR_IMAGE_HEIGHT = 30;

    List<PossiblePlate> detectCharsInPlates(List<PossiblePlate> possiblePlates) {
        if (possiblePlates.isEmpty()) return possiblePlates;

        // at least one plate
        for (PossiblePlate possiblePlate : possiblePlates) {
            // get grayscale and threshold images
            Preprocess.preprocess(possiblePlate.getImgPlate(), possiblePlate.getImgGrayscale(),
                    possiblePlate.getImgThresh());

            // Upscale by 60% for better viewing and character recognition
            Imgproc.resize(possiblePlate.getImgThresh(), possiblePlate.getImgThresh(), new Size(), 1.6, 1.6);

            // Threshold again to eliminate any gray areas
            Imgproc.threshold(possiblePlate.getImgThresh(), possiblePlate.getImgThresh(),
                    0.0, 255.0, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

            // Find possible chars in the plate
            List<PossibleChar> possibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.getImgThresh());

            // Find groups of matching chars within the plate
            List<List<PossibleChar>> matchingCharsInPlate = findMatchingChars(possibleCharsInPlate);

            // No groups of matching chars were found in the plate
            if (matchingCharsInPlate.size() == 0) {
                possiblePlate.setStrChars("");
                continue;
            }

            // Suppose the longest list of matching chars is the correct one
            int longestLen = 0;
            int longestLendIdx = 0;

            for (int i = 0; i < matchingCharsInPlate.size(); i++) {
                if (matchingCharsInPlate.get(i).size() > longestLen) {
                    longestLen = matchingCharsInPlate.get(i).size();
                    longestLendIdx = i;
                }
            }
            List<PossibleChar> longestMatchingChars = matchingCharsInPlate.get(longestLendIdx);

            // Char recognition on the longest list
            possiblePlate.setStrChars(recognizeCharsInPlate(possiblePlate.getImgThresh(), longestMatchingChars));
        }

        return possiblePlates;
    }

    private List<PossibleChar> findPossibleCharsInPlate(Mat imgThresh) {
        List<PossibleChar> listOfPossibleChars = new ArrayList<>();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat imgThreshCopy = imgThresh.clone();

        // Find all contours in plate
        Imgproc.findContours(imgThreshCopy, contours, imgThreshCopy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        for (MatOfPoint contour : contours) {
            PossibleChar possibleChar = new PossibleChar(contour);

            if (checkIfPossibleChar(possibleChar))
                listOfPossibleChars.add(possibleChar);
        }

        return listOfPossibleChars;
    }

    boolean checkIfPossibleChar(PossibleChar possibleChar) {
        // Rough check on a contour to see if it could be a char,
        return possibleChar.getBoundingRect().area() > MIN_PIXEL_AREA &&
                possibleChar.getBoundingRect().width > MIN_PIXEL_WIDTH &&
                possibleChar.getBoundingRect().height > MIN_PIXEL_HEIGHT &&
                possibleChar.getDblAspectRatio() > MIN_ASPECT_RATIO &&
                possibleChar.getDblAspectRatio() < MAX_ASPECT_RATIO;
    }

    List<List<PossibleChar>> findMatchingChars(List<PossibleChar> possibleCharsInPlate) {
        // Re-arrange chars into a list of lists of matching chars
        List<List<PossibleChar>> matchingCharsInPlate = new ArrayList<>();

        for (PossibleChar possibleChar : possibleCharsInPlate) {
            List<PossibleChar> listOfMatchingChars = findListOfMatchingChars(possibleChar, possibleCharsInPlate);
            listOfMatchingChars.add(possibleChar);

            if (listOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS)
                continue;

            matchingCharsInPlate.add(listOfMatchingChars);

            List<PossibleChar> listWithCurrentMatchesRemoved = new ArrayList<>();
            for (PossibleChar possChar : possibleCharsInPlate) {
                if (listOfMatchingChars.indexOf(possChar) == listOfMatchingChars.size() - 1)
                    listWithCurrentMatchesRemoved.add(possChar);
            }

            List<List<PossibleChar>> recursiveListOfMatchingChars;
            // recursive call
            recursiveListOfMatchingChars = findMatchingChars(listWithCurrentMatchesRemoved);

            matchingCharsInPlate.addAll(recursiveListOfMatchingChars);
            break;
        }

        return matchingCharsInPlate;
    }

    private List<PossibleChar> findListOfMatchingChars(PossibleChar possibleChar, List<PossibleChar> listOfChars) {
        // Given a possible char and list possible chars, find matching chars
        List<PossibleChar> listOfMatchingChars = new ArrayList<>();

        for (PossibleChar possibleMatchingChar : listOfChars) {
            if (possibleMatchingChar == possibleChar)
                continue;

            double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);
            double dblAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar);
            double dblChangeInArea = Math.abs(possibleMatchingChar.getBoundingRect().area() - possibleChar.getBoundingRect().area()) / possibleChar.getBoundingRect().area();
            double dblChangeInWidth = Math.abs(possibleMatchingChar.getBoundingRect().width - possibleChar.getBoundingRect().width) / (double) possibleChar.getBoundingRect().width;
            double dblChangeInHeight = Math.abs(possibleMatchingChar.getBoundingRect().height - possibleChar.getBoundingRect().height) / (double) possibleChar.getBoundingRect().height;

            // Check if chars match
            if (dblDistanceBetweenChars < (possibleChar.getDblDiagonalSize() * MAX_DIAG_SIZE_MULTIPLE_AWAY) &&
                    dblAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS &&
                    dblChangeInArea < MAX_CHANGE_IN_AREA &&
                    dblChangeInWidth < MAX_CHANGE_IN_WIDTH &&
                    dblChangeInHeight < MAX_CHANGE_IN_HEIGHT) {
                listOfMatchingChars.add(possibleMatchingChar);
            }
        }

        return listOfMatchingChars;
    }

    double distanceBetweenChars(PossibleChar firstChar, PossibleChar secondChar) {
        int intX = Math.abs(firstChar.getIntCenterX() - secondChar.getIntCenterX());
        int intY = Math.abs(firstChar.getIntCenterY() - secondChar.getIntCenterY());

        return Math.sqrt(Math.pow(intX, 2) + Math.pow(intY, 2));
    }

    private double angleBetweenChars(PossibleChar firstChar, PossibleChar secondChar) {
        double dblAdj = Math.abs(firstChar.getIntCenterX() - secondChar.getIntCenterX());
        double dblOpp = Math.abs(firstChar.getIntCenterY() - secondChar.getIntCenterY());

        double dblAngleInRad = Math.atan(dblOpp / dblAdj);

        return dblAngleInRad * (180.0 / Math.PI);
    }

    private String recognizeCharsInPlate(Mat imgThresh, List<PossibleChar> listOfMatchingChars) {
        StringBuilder strChars = new StringBuilder();

        // sort chars from left to right
        Collections.sort(listOfMatchingChars);

        Mat imgThreshColor = new Mat();
        Imgproc.cvtColor(imgThresh, imgThreshColor, Imgproc.COLOR_GRAY2BGR);

        for (PossibleChar currentChar : listOfMatchingChars) {
            // draw green box around the char
            Imgproc.rectangle(imgThreshColor, currentChar.getBoundingRect().br(), currentChar.getBoundingRect().tl(), new Scalar(0, 255, 0, 128), 2);

            Mat imgROItoBeCloned = new Mat(imgThresh, currentChar.getBoundingRect());
            Mat imgROI = imgROItoBeCloned.clone();

            Mat imgROIResized = new Mat();
            // resize image for char recognition
            Imgproc.resize(imgROI, imgROIResized, new Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));

            Mat matROIFloat = new Mat();
            // convert Mat to float, necessary for call to findNearest
            imgROIResized.convertTo(matROIFloat, CvType.CV_32FC1);

            Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);
            Mat matCurrentChar = new Mat(0, 0, CvType.CV_32F);

            MainActivity.kNearest.findNearest(matROIFlattenedFloat, 1, matCurrentChar);

            float fltCurrentChar = (float) matCurrentChar.get(0, 0)[0];

            strChars.append((char) ((int) fltCurrentChar));
        }
        return strChars.toString();
    }
}
