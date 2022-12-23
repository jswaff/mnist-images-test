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
        System.out.println("executing mnist testapp");

        List<Image> trainingImages = ImageImporter.importImages(true);
        System.out.println("loaded " + trainingImages.size() + " training images");

        List<Image> testImages = ImageImporter.importImages(false);
        System.out.println("loaded " + testImages.size() + " test images");

        // create a network
        Network network = Network.builder()
                .numInputUnits(28 * 28)
                .layers(List.of(
                        new Layer(50, new Sigmoid()),
                        //new Layer(25, new Sigmoid()),
                        new Layer(10, new Identity()) // TODO: need a softmax layer
                ))
                .costFunction(new MSE())
                .build();
        network.initialize();

        // load the test data
        Pair<SimpleMatrix, SimpleMatrix> X_Y_test = loadXY(testImages);
        SimpleMatrix X_test = X_Y_test.getValue0();
        SimpleMatrix Y_test = X_Y_test.getValue1();
        System.out.println("Y_test rows: " + Y_test.numRows());
        System.out.println("Y_test cols: " + Y_test.numCols());

        // get the initial cost
        SimpleMatrix P_init = network.predict(X_test);
        System.out.println("initial cost: " + network.cost(P_init, Y_test));

        Pair<SimpleMatrix, SimpleMatrix> X_Y_train = loadXY(trainingImages);
        SimpleMatrix X_train = X_Y_train.getValue0();
        SimpleMatrix Y_train = X_Y_train.getValue1();

        network.train(X_train, Y_train, 100);

        SimpleMatrix P_final = network.predict(X_test);
        System.out.println("\tfinal cost: " + network.cost(P_final, Y_test));

        // measure accuracy
        int totalCorrect= 0;
        System.out.println("P_final rows: " + P_final.numRows());
        System.out.println("P_final cols: " + P_final.numCols());
        for (int c=0;c<P_final.numCols();c++) {
            double label = getLabel(Y_test, c);
            double predictedLabel = getLabel(P_final, c);

            if (Math.abs(label - predictedLabel) < 0.001) {
                totalCorrect++;
            }
        }
        System.out.println("accuracy: " + totalCorrect + " / " + testImages.size() + " (" +
                totalCorrect *100.0/testImages.size() + "%)");

        System.out.println("execution complete.  bye.");
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

            // sanity check
            if (getLabel(Y, c) != label) {
                throw new RuntimeException("Something is jacked up.");
            }
        }

        return new Pair<>(X, Y);
    }

    // TODO: this wont' be needed once a softmax activation is in place
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
}
