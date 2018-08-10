// Global variables so I remember them
config_name = null;
img_idx = null;

img = null;
dets = null;

colors = ['#FF0000', '#FF7F00', '#00FF00', '#0000FF', '#4B0082', '#9400D3'];

settings = {
	'top_k': 5,
	'font_height': 20,
	'display_class': true,
	'display_score': true,
	'display_bbox': true,
	'display_mask': true,
}

$.urlParam = function(name){
	var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);
	if (results==null){
		return null;
	}
	else{
		return decodeURI(results[1]) || 0;
	}
}

$(document).ready(function() {
	config_name = $.urlParam('config');
	$('#config_name').html(config_name);

	img_idx = $.urlParam('idx');
	if (img_idx === null) img_idx = 0;
	img_idx = parseInt(img_idx);
	
	$.getJSON('dets/' + config_name + '.json', function(data) {
		img_idx = (img_idx+data.images.length) % data.images.length;
		var info = data.info;
		var data = data.images[img_idx];

		// These are globals on purpose
		dets = data.dets;
		img = new Image();

		img.onload = function() { render(); }
		img.src = 'image' + data.image_id;

		$('#image_name').html(data.image_id);
		$('#image_idx').html(img_idx);

		fill_info(info);
		fill_controls();
	});
});

function is_object(val) { return val === Object(val); }

function fill_info(info) {
	var html = '';
	
	var add_item = function(item, val) {
		html += '<span class="info_item">' + item + '</span>'
		html += '&nbsp;'
		html += '<span class="info_value">' + val + '</span>'
		html += '<br />'
	}

	for (var item in info) {
		var val = info[item];

		if (is_object(val)) {
			html += '<span class="info_section">' + item + '</span><br />';

			for (var item2 in val)
				add_item(item2, val[item2]);

			html += '<br />'
		} else add_item(item, val);
	}
	
	$('#info_box').html(html);
}

function fill_controls() {
	html = '<br>';
	html += '<a href="viewer.html?config=' + config_name + '&idx=' + (img_idx-1) +'">Prev</a>';
	html += '&nbsp;&nbsp;&nbsp;';
	html += '<a href="viewer.html?config=' + config_name + '&idx=' + (img_idx+1) +'">Next</a>';
	html += '<br><br><br>';
	html += '<a href="/">Back</a>';
	$('#control_box').html(html);
}

function render() {
	var canvas = document.querySelector('#image_canvas');
	var ctx = canvas.getContext('2d');

	canvas.style.width='100%';
	canvas.style.height='95%';
	canvas.width  = canvas.offsetWidth;
	canvas.height = canvas.offsetHeight;

	var scale = Math.min(canvas.width / img.width, canvas.height / img.height);

	var im_x = canvas.width/2-img.width*scale/2;
	var im_y = canvas.height/2-img.height*scale/2;
	ctx.translate(im_x, im_y);
	ctx.drawImage(img, 0, 0, img.width * scale, img.height * scale);

	for (var i = Math.min(dets.length, settings.top_k)-1; i >= 0 ; i--) {
		ctx.strokeStyle = colors[i % colors.length];
		ctx.fillStyle   = ctx.strokeStyle;
		ctx.lineWidth   = 4;
		ctx.font = settings.font_height + 'px sans-serif';

		var x = dets[i].bbox[0] * scale;
		var y = dets[i].bbox[1] * scale;
		var w = dets[i].bbox[2] * scale;
		var h = dets[i].bbox[3] * scale;

		if (settings.display_bbox) {
			ctx.strokeRect(x, y, w, h);
			ctx.stroke();
		}

		var text_array = []
		if (settings.display_class)
			text_array.push(dets[i].category);
		if (settings.display_score)
			text_array.push(Math.round(dets[i].score * 1000) / 1000);

		if (text_array.length > 0) {
			var text = text_array.join(' ');

			text_w = ctx.measureText(text).width;
			ctx.fillRect(x-ctx.lineWidth/2, y-settings.font_height-8, text_w+ctx.lineWidth, settings.font_height+8);
			
			ctx.fillStyle = 'white';
			ctx.fillText(text, x, y-8);
		}
	}
}
