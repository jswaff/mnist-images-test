package com.jamesswafford.ml.mnisttestapp;

import lombok.Data;

import java.util.Arrays;

@Data
public class Image {

    private final double[] data;
    private final int label;

    public Image(double[] data, int label) {
        this.data = Arrays.copyOf(data, data.length);
        this.label = label;
    }

    private char toChar(double val) {
        return " .:-=+*#%@".charAt(Math.min((int) (val * 10), 9));
    }

    @Override
    public String toString() {

        StringBuilder sb = new StringBuilder();
        sb.append("Label: ").append(label);

        sb.append("\nImage:\n");
        for (int i=0;i< data.length;i++) {
            sb.append(toChar(data[i]));
            if (i%28==0) {
                sb.append("\n");
            }
        }

        return sb.toString();
    }
}
