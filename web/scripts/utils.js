function load_RLE(rle_obj, fillColor=[255, 255, 255], alpha=255) {
	var h = rle_obj.size[0], w = rle_obj.size[1];
	var counts = uncompress_RLE(rle_obj.counts);
	
	var buffer_size = (w*h*4);
	var buffer = new Uint8ClampedArray(w*h*4);
	var bufferIdx = 0;

	for (var countsIdx = 0; countsIdx < counts.length; countsIdx++) {
		while (counts[countsIdx] > 0) {
			// Kind of transpose the image as we go
			if (bufferIdx >= buffer_size)
				bufferIdx = (bufferIdx % buffer_size) + 4;
			
			buffer[bufferIdx+0] = fillColor[0];
			buffer[bufferIdx+1] = fillColor[1];
			buffer[bufferIdx+2] = fillColor[2];
			buffer[bufferIdx+3] = alpha * (countsIdx % 2);

			bufferIdx += 4*w;
			counts[countsIdx]--;
		}
	}

	// Load into an off-screen canvas and return an image with that data
	var canvas = document.createElement('canvas');
	var ctx = canvas.getContext('2d');

	canvas.width = w;
	canvas.height = h;

	var idata = ctx.createImageData(w, h);
	idata.data.set(buffer);

	ctx.putImageData(idata, 0, 0);

	var img = new Image();
	img.src = canvas.toDataURL();

	return img;
}

function uncompress_RLE(rle_str) {
	// Don't ask me how this works--I'm just transcribing from the pycocotools c api.
	var p = 0, m = 0;
	var counts = Array(rle_str.lenght);

	while (p < rle_str.length) {
		var x=0, k=0, more=1;

		while (more) {
			var c = rle_str.charCodeAt(p) - 48;
			x |= (c & 0x1f) << 5*k;
			more = c & 0x20;
			p++; k++;
			if (!more && (c & 0x10))
				x |= (-1 << 5*k);
		}

		if (m > 2)
			x += counts[m-2];
		counts[m++] = (x >>> 0);
	}

	return counts;
}

function hexToRgb(hex) {
	var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
	return result ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)] : null;
}
