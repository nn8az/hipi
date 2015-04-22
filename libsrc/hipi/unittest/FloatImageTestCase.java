package hipi.unittest;

import static org.junit.Assert.assertEquals;
import hipi.image.FloatImage;
import hipi.image.io.ImageDecoder;
import hipi.image.io.PPMImageUtil;
import hipi.util.ByteUtils;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.util.NoSuchElementException;
import java.util.Scanner;

import org.junit.Test;

public class FloatImageTestCase {

	@Test
	public void testFloatImageWritable() throws IOException {
		ImageDecoder decoder = PPMImageUtil.getInstance();
		FileInputStream fis;
		String[] fileName = { "canon-ixus", "cmyk-jpeg-format" };
		for (int i = 0; i < fileName.length; i++) {
			fis = new FileInputStream("data/test/JPEGImageUtilTestCase/truth/"
					+ fileName[i] + ".ppm");
			FloatImage image = decoder.decodeImage(fis);
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			image.write(new DataOutputStream(bos));
			ByteArrayInputStream bis = new ByteArrayInputStream(
					bos.toByteArray());
			FloatImage newImage = new FloatImage();
			newImage.readFields(new DataInputStream(bis));
			assertEquals(fileName[i] + " writable test fails", image, newImage);
		}
	}

	@Test
	public void testGaussianFilter() throws IOException {
		// Setup the ground-truth file for a read.
		FileInputStream truthFile = new FileInputStream(
				"data/test/FloatImageTestCase/gaussianFilter/gaussian.txt");

		// Loop through each trial.
		trialLoop: while (true) {
			// Read the header.
			String header = "";
			int c;
			while (true) {
				c = truthFile.read();
				if (c == -1) {
					break trialLoop;
				} else if (c == '\r') {
					truthFile.read();
					break;
				} else {
					header += (char) c;
				}
			}

			// Parses the information in the header.
			int radius = Integer.parseInt(header);
			int dimension = 2 * radius + 1;

			// Read the data.
			byte[] buffer = new byte[dimension * dimension * 4];
			truthFile.read(buffer);

			// Construct a FloatImage representation of the ground-truth
			// gaussian filter.
			ByteBuffer bodyReader = ByteBuffer.wrap(buffer);
			float[] pels = new float[dimension * dimension];
			for (int i = 0; i < pels.length; i++) {
				pels[i] = bodyReader.getFloat();
			}
			FloatImage truthImg = new FloatImage(dimension, dimension, 1, pels);

			// Test the gaussianFilter().
			FloatImage testImg = FloatImage.gaussianFilter(radius);
			assertEquals("gaussianFilter() fails for radius " + radius,
					testImg, truthImg);

			// Skip the end-of-trial line break
			truthFile.skip(2);
		}

	}
}
