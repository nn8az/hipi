__kernel void convolveKernel(__global const float *srcPels, __global const int *srcHeader,
	__global const float *filterPels, __global const int *filterHeader, __global float *destPels) {
	int id = get_global_id(0);
	
	int srcWidth = srcHeader[0];
	int srcHeight = srcHeader[1];
	int srcBands = srcHeader[2];
	int filterWidth = filterHeader[0];
	int filterHeight = filterHeader[1];
	int filterBands = filterHeader[2];
	
	float totalValue = 0.0f;
	for (int filterY = 0; filterY < filterHeight; filterY++) {
		for (int filterX = 0; filterX < filterWidth; filterX++) {
			int dx = filterX - filterWidth / 2;
			int dy = filterY - filterHeight / 2;
			
			int srcC = id % srcBands;
			int srcX = (id / srcBands) % srcWidth;
			int srcY = id / (srcBands * srcWidth);
			srcX += dx;
			srcY += dy;
			
			if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight) {
				continue;
			}
			
			int filterC = srcC % filterBands;
			
			int srcIndex = srcC + (srcX + srcY * srcWidth) * srcBands;
			int filterIndex = filterC + (filterX + filterY * filterWidth) * filterBands;
			totalValue += filterPels[filterIndex] * srcPels[srcIndex];
		}
	}
	destPels[id] = totalValue;
}