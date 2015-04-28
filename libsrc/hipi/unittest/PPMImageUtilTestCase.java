package hipi.unittest;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import hipi.image.FloatImage;
import hipi.image.io.PPMImageUtil;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.junit.Test;

public class PPMImageUtilTestCase {

	@Test
	public void testSmallEncodeImage() throws IOException {
		// hardcoded 2x2 "image" to write
		float[] pels = {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0};
		FloatImage img = new FloatImage(2, 2, 3, pels);
		
		// write the image
		PPMImageUtil io = PPMImageUtil.getInstance();
		FileOutputStream fos = new FileOutputStream("data/test/PPMImageUtilTestCase/encode/block.ppm");
		io.encodeImage(img, null, fos);
		
		// read in the written image as binary
		FileInputStream fis = new FileInputStream("data/test/PPMImageUtilTestCase/encode/block.ppm");
		byte[] testBuffer = new byte[32];
		int byteRead = fis.read(testBuffer);
		assertEquals(byteRead, 23);
		
		// read in the ground truth as binary
		fis = new FileInputStream("data/test/PPMImageUtilTestCase/truth/block.ppm");
		byte[] truthBuffer = new byte[32];
		int gtByteRead = fis.read(truthBuffer);
		assertEquals(gtByteRead, 23);
		assertArrayEquals(testBuffer, truthBuffer);
	}

}
