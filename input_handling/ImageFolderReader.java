package input_handling;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import javax.imageio.ImageIO;

public class ImageFolderReader {

    private static final Set<String> IMAGE_EXTENSIONS = Set.of("png", "jpg", "jpeg", "bmp", "gif");

    public List<BufferedImage> readImagesFromFolder(String folderPath, boolean recursive) throws IOException {
        Path folder = Paths.get(folderPath);
        if (!Files.isDirectory(folder)) {
            throw new IOException("Path is not a directory: " + folderPath);
        }

        List<BufferedImage> images = new ArrayList<>();

        if (recursive) {
            try (var stream = Files.walk(folder)) {
                stream.filter(Files::isRegularFile)
                        .filter(this::hasImageExtension)
                        .forEach(path -> addImageSafely(path, images));
            }
        } else {
            try (var stream = Files.newDirectoryStream(folder)) {
                for (Path path : stream) {
                    if (Files.isRegularFile(path) && hasImageExtension(path)) {
                        addImageSafely(path, images);
                    }
                }
            }
        }
        return images;
    }

    private boolean hasImageExtension(Path path) {
        String fileName = path.getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        if (dot == -1) return false;
        String ext = fileName.substring(dot + 1).toLowerCase();
        return IMAGE_EXTENSIONS.contains(ext);
    }

    private void addImageSafely(Path path, List<BufferedImage> images) {
        try {
            BufferedImage img = ImageIO.read(path.toFile());
            if (img != null) {
                images.add(img);
            } else {
                System.err.println("Warning: Could not decode image (unsupported format?) - " + path);
            }
        } catch (IOException e) {
            System.err.println("Warning: Error reading image " + path + " - " + e.getMessage());
        }
    }
}