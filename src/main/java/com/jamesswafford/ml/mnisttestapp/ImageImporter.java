package com.jamesswafford.ml.mnisttestapp;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class ImageImporter {

    public static List<Image> importImages(boolean train) {
        List<Image> images = new ArrayList<>();

        ClassLoader classLoader = ImageImporter.class.getClassLoader();

        String labelsFileName = train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte";
        String imagesFileName = train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte";


        try (
            DataInputStream labelsIS = new DataInputStream(
                    Objects.requireNonNull(classLoader.getResourceAsStream(labelsFileName)));
            DataInputStream imagesIS = new DataInputStream(
                    Objects.requireNonNull(classLoader.getResourceAsStream(imagesFileName)));
        )
        {
            if (imagesIS.readInt() != 2051) {
                throw new IOException("Unknown file format for images");
            }

            if (labelsIS.readInt() != 2049) {
                throw new IOException("Unknown file format for labels");
            }

            int numImages = imagesIS.readInt();
            int numLabels = labelsIS.readInt();

            if (numLabels != numImages) {
                throw new IllegalStateException("Number of labels " + numLabels + " does not match number of images " + numImages);
            }

            int numRows = imagesIS.readInt();
            int numCols = imagesIS.readInt();
            byte[] data = new byte[numRows * numCols];

            for (int i=0;i<numImages;i++) {
                int numRead = imagesIS.read(data, 0, data.length);
                if (numRead != data.length) {
                    throw new IOException("Expected to read " + data.length + " bytes. actually read " + numRead);
                }
                double[] img = new double[data.length];
                for (int j=0;j<data.length;j++) {
                    img[j] = (data[j] & 255) / 255.0;
                }
                int label = labelsIS.readByte();
                images.add(new Image(img, label));
            }

            return images;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
