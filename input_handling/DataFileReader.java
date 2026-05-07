package input_handling;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;


public class DataFileReader {

    public List<String[]> readDataFile(String filePath, String delimiter) throws IOException {
        List<String[]> records = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(Paths.get(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] values = line.split(delimiter);
                records.add(values);
            }
        }
        return records;
    }

    public List<String[]> readDataFile(String filePath) throws IOException {
        return readDataFile(filePath, "\\s+");
    }
}