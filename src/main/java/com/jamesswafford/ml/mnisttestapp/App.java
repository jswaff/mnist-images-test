package com.jamesswafford.ml.mnisttestapp;

import com.jamesswafford.ml.nn.Layer;
import com.jamesswafford.ml.nn.Network;
import com.jamesswafford.ml.nn.activation.Identity;
import com.jamesswafford.ml.nn.activation.Sigmoid;
import com.jamesswafford.ml.nn.cost.MSE;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import java.util.Collections;
import java.util.List;

public class App {

    public static void main(String[] args) {
        System.out.println("executing mnist testapp\n");

        List<Image> trainingImages = ImageImporter.importImages(true);
        System.out.println("loaded " + trainingImages.size() + " training images");

        List<Image> testImages = ImageImporter.importImages(false);
        System.out.println("loaded " + testImages.size() + " test images");

        // create a network
        Network network = Network.builder()
                .numInputUnits(28 * 28)
                .layers(List.of(
                        new Layer(38, Sigmoid.INSTANCE),
                        new Layer(14, Sigmoid.INSTANCE),
                        new Layer(10, Identity.INSTANCE) // TODO: softmax
                ))
                .costFunction(MSE.INSTANCE)
                .build();
        network.initialize();

        // load the test data
        //Pair<SimpleMatrix, SimpleMatrix> X_Y_test = loadXY(testImages);
        //SimpleMatrix X_test = X_Y_test.getValue0();
        //SimpleMatrix Y_test = X_Y_test.getValue1();
        Pair<double[][], double[][]> X_Y_test = loadXY2(testImages);
        double[][] X_test = X_Y_test.getValue0();
        double[][] Y_test = X_Y_test.getValue1();

        // get the initial cost
        //SimpleMatrix P_init = network.predict(X_test);
        double[][] P_init = network.predict(X_test);
        System.out.println("initial cost: " + network.cost(P_init, Y_test));

//        Pair<SimpleMatrix, SimpleMatrix> X_Y_train = loadXY(trainingImages);
//        SimpleMatrix X_train = X_Y_train.getValue0();
//        SimpleMatrix Y_train = X_Y_train.getValue1();
        Pair<double[][], double[][]> X_Y_train = loadXY2(trainingImages);
        double[][] X_train = X_Y_train.getValue0();
        double[][] Y_train = X_Y_train.getValue1();

        long startTime = System.currentTimeMillis();
        network.train(X_train, Y_train, 50, 10, 3.0, X_test, Y_test);
        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("elapsed time (sec): " + elapsed/1000);
        //SimpleMatrix P_final = network.predict(X_test);
        double[][] P_final = network.predict(X_test);
        System.out.println("final cost: " + network.cost(P_final, Y_test));

        // measure accuracy
        int totalCorrect= 0;
        for (int c=0;c<P_final[0].length;c++) {
        //for (int c=0;c<P_final.numCols();c++) {
            double label = getLabel2(Y_test, c);
            double predictedLabel = getLabel2(P_final, c);

            if (Math.abs(label - predictedLabel) < 0.001) {
                totalCorrect++;
            }
        }

        System.out.println("accuracy: " + totalCorrect + " / " + testImages.size() + " (" +
                totalCorrect *100.0/testImages.size() + "%)");

        System.out.println("\nexecution complete.  bye.");
    }

    private static Pair<SimpleMatrix, SimpleMatrix> loadXY(List<Image> images) {
        Collections.shuffle(images);

        SimpleMatrix X = new SimpleMatrix(28 * 28, images.size());
        SimpleMatrix Y = new SimpleMatrix(10, images.size());

        // one image per column
        for (int c=0;c<images.size();c++) {
            Image image = images.get(c);

            // set the input features for this training example
            double[] data = image.getData();
            for (int r = 0; r < X.numRows(); r++) {
                X.set(r, c, data[r]);
            }

            // set label
            int label = image.getLabel();
            if (label < 0 || label > 9) {
                throw new IllegalStateException("Invalid label: " + label);
            }
            for (int r=0;r<10;r++) {
                Y.set(r, c, r==label ? 1.0 : 0.0);
            }
        }

        return new Pair<>(X, Y);
    }

    private static Pair<double[][], double[][]> loadXY2(List<Image> images) {
        Collections.shuffle(images);

        double[][] X = new double[28*28][images.size()];
        double[][] Y = new double[10][images.size()];

        // one image per column
        for (int c=0;c<images.size();c++) {
            Image image = images.get(c);

            // set the input features for this training example
            double[] data = image.getData();
            for (int r = 0; r < X.length; r++) {
                X[r][c] = data[r];
            }

            // set label
            int label = image.getLabel();
            if (label < 0 || label > 9) {
                throw new IllegalStateException("Invalid label: " + label);
            }
            for (int r=0;r<10;r++) {
                Y[r][c] = (r==label ? 1.0 : 0.0);
            }
        }

        return new Pair<>(X,Y);
    }

    private static int getLabel(SimpleMatrix P, int col) {
        double biggestVal = P.get(0, col);
        int biggestInd = 0;
        for (int r=1;r<P.numRows();r++) {
            double val = P.get(r, col);
            if (val > biggestVal) {
                biggestVal = val;
                biggestInd = r;
            }
        }
        return biggestInd;
    }

    private static int getLabel2(double[][] P, int col) {
        double biggestVal = P[0][col];
        int biggestInd = 0;
        for (int r=1;r<P.length;r++) {
            double val = P[r][col];
            if (val > biggestVal) {
                biggestVal = val;
                biggestInd = r;
            }
        }
        return biggestInd;
    }
}
