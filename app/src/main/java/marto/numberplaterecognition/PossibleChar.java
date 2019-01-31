package marto.numberplaterecognition;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public class PossibleChar implements Comparable<PossibleChar> {

    private List<Point> contour;
    private Rect boundingRect;
    private int intCenterX;
    private int intCenterY;
    private double dblDiagonalSize;
    private double dblAspectRatio;

    public PossibleChar(MatOfPoint contour) {
        this.contour = contour.toList();

        boundingRect = Imgproc.boundingRect(contour);

        intCenterX = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
        intCenterY = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;

        dblDiagonalSize = Math.sqrt(Math.pow(boundingRect.width, 2) + Math.pow(boundingRect.height, 2));
        dblAspectRatio = (float)boundingRect.width / (float)boundingRect.height;
    }

    @Override
    public int compareTo(PossibleChar possibleChar) {
        return possibleChar.intCenterX - this.intCenterX;
    }

    public List<Point> getContour() {
        return contour;
    }

    public void setContour(List<Point> contour) {
        this.contour = contour;
    }

    public Rect getBoundingRect() {
        return boundingRect;
    }

    public void setBoundingRect(Rect boundingRect) {
        this.boundingRect = boundingRect;
    }

    public int getIntCenterX() {
        return intCenterX;
    }

    public void setIntCenterX(int intCenterX) {
        this.intCenterX = intCenterX;
    }

    public int getIntCenterY() {
        return intCenterY;
    }

    public void setIntCenterY(int intCenterY) {
        this.intCenterY = intCenterY;
    }

    public double getDblDiagonalSize() {
        return dblDiagonalSize;
    }

    public void setDblDiagonalSize(double dblDiagonalSize) {
        this.dblDiagonalSize = dblDiagonalSize;
    }

    public double getDblAspectRatio() {
        return dblAspectRatio;
    }

    public void setDblAspectRatio(double dblAspectRatio) {
        this.dblAspectRatio = dblAspectRatio;
    }
}
