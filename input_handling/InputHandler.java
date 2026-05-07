package input_handling;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;

import input_handling.CsvReader;
import input_handling.DataFileReader;
import input_handling.ImageFolderReader;


public class InputHandler {

    private final CsvReader csvReader;
    private final DataFileReader dataFileReader;
    private final ImageFolderReader imageFolderReader;

    public InputHandler() {
        this.csvReader = new CsvReader();
        this.dataFileReader = new DataFileReader();
        this.imageFolderReader = new ImageFolderReader();
    }


    public Object read(String type, String path, boolean recursive) throws IOException {
        switch (type.toLowerCase()) {
            case "csv":
                return csvReader.readCSV(path);
            case "data":
                return dataFileReader.readDataFile(path);
            case "images":
                return imageFolderReader.readImagesFromFolder(path, recursive);
            default:
                throw new IllegalArgumentException("Unknown type: " + type + ". Use 'csv', 'data', or 'images'.");
        }
    }

    public Object autoRead(String path, boolean recursiveForImages) throws IOException {
        java.nio.file.Path p = java.nio.file.Paths.get(path);
        if (java.nio.file.Files.isDirectory(p)) {
            return imageFolderReader.readImagesFromFolder(path, recursiveForImages);
        } else {
            String name = p.getFileName().toString().toLowerCase();
            if (name.endsWith(".csv")) {
                return csvReader.readCSV(path);
            } else if (name.endsWith(".data")) {
                return dataFileReader.readDataFile(path);
            } else {
                throw new IllegalArgumentException("Unsupported file type. Use .csv or .data");
            }
        }
    }

    // Simple interactive demonstration
    public static void main(String[] args) {
        InputHandler handler = new InputHandler();
        Scanner scanner = new Scanner(System.in);

        System.out.println("Choose reader type: csv / data / images");
        String type = scanner.nextLine().trim();

        System.out.print("Enter path: ");
        String path = scanner.nextLine().trim();

        try {
            if (type.equalsIgnoreCase("images")) {
                System.out.print("Recursive search? (true/false): ");
                boolean recursive = Boolean.parseBoolean(scanner.nextLine().trim());
                var images = (java.util.List<java.awt.image.BufferedImage>) handler.read(type, path, recursive);
                System.out.println("Loaded " + images.size() + " images.");
                if (!images.isEmpty()) {
                    System.out.println("First image dimensions: " + images.get(0).getWidth() + "x" + images.get(0).getHeight());
                }
            } else {
                var data = (java.util.List<String[]>) handler.read(type, path, false);
                System.out.println("First 5 rows:");
                for (int i = 0; i < Math.min(5, data.size()); i++) {
                    System.out.println(java.util.Arrays.toString(data.get(i)));
                }
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
        scanner.close();
    }
}