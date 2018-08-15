// Global variables so I remember them
config_name = null;
img_idx = null;

img = null;
dets = null;
masks = null;

// Must be in hex
colors = ['#FF0000', '#FF7F00', '#00FF00', '#0000FF', '#4B0082', '#9400D3'];

settings = {
	'top_k': 5,
	'font_height': 20,
	'mask_alpha': 100,

	'show_class': true,
	'show_score': true,
	'show_bbox': true,
	'show_mask': true,
	
	'show_one': false,
}

function save_settings() {
	Cookies.set('settings', settings);
}

function load_settings() {
	var new_settings = Cookies.getJSON('settings');

	for (var key in new_settings)
		settings[key] = new_settings[key];
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
	
	load_settings();

	$.getJSON('dets/' + config_name + '.json', function(data) {
		img_idx = (img_idx+data.images.length) % data.images.length;
		var info = data.info;
		var data = data.images[img_idx];

		// These are globals on purpose
		dets = data.dets;
		img = new Image();
		masks = Array(dets.length);

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
	var html = '';

	var append_html = function() {
		$('#control_box').append(html);
		html = '';
	}

	var make_slider = function (name, setting, min, max) {
		settings[setting] = Math.min(max, settings[setting]);
		var value = settings[setting];

		html += '<div class="setting">';
		html += '<span class="setting_label">' + name + '</span>';
		html += '<input type="range" min="' + min + '" max="' + max + '" value="' + value + '" id="' + setting + '" class="setting_input">';
		html += '<span class="setting_value", id="' + setting + '">' + value + '</span>';
		html += '</div>';
		append_html();

		$('input#'+setting).change(function(e) {
			settings[setting] = $('input#'+setting).prop('value');
			$('span#'+setting).html(settings[setting]);
			save_settings();
			render();
		});
	}

	var make_toggle = function(name, setting) {
		html += '<div class="setting" style="grid-template-columns: 1fr 0px 50px;">';
		html += '<span class="setting_label">' + name + '</span>';
		html += '<label class="switch">';
		html += '<input type="checkbox" id="' + setting + '" class="setting_input"' + (settings[setting] ? 'checked' : '') + '>';
		html += '<span class="slider round"></span>';
		html += '</label></div>';
		append_html();

		$('input#' + setting).change(function (e) {
			settings[setting] = $('input#' + setting).prop('checked');
			save_settings();
			render();
		});
	}

	
	make_slider('Top K', 'top_k', 1, dets.length);
	make_toggle('Show One', 'show_one');
	html += '<br>';
	make_toggle('Show BBox', 'show_bbox');
	make_toggle('Show Class', 'show_class');
	make_toggle('Show Score', 'show_score');
	html += '<br>';
	make_slider('Mask Alpha', 'mask_alpha', 0, 255);
	make_toggle('Show Mask', 'show_mask');

	html += '<br><br>';
	html += '<a href="viewer.html?config=' + config_name + '&idx=' + (img_idx-1) +'">Prev</a>';
	html += '&nbsp;&nbsp;&nbsp;';
	html += '<a href="viewer.html?config=' + config_name + '&idx=' + (img_idx+1) +'">Next</a>';
	html += '<br><br>';
	html += '<a href="/">Back</a>';

	append_html();
}

function render() {
	var canvas = document.querySelector('#image_canvas');
	var ctx = canvas.getContext('2d');

	canvas.style.width='100%';
	canvas.style.height='94%';
	canvas.width  = canvas.offsetWidth;
	canvas.height = canvas.offsetHeight;

	var scale = Math.min(canvas.width / img.width, canvas.height / img.height);

	var im_x = canvas.width/2-img.width*scale/2;
	var im_y = canvas.height/2-img.height*scale/2;
	ctx.translate(im_x, im_y);
	ctx.drawImage(img, 0, 0, img.width * scale, img.height * scale);

	var startIdx = Math.min(dets.length, settings.top_k)-1;
	var endIdx   = (settings.show_one ? startIdx : 0);

	for (var i = startIdx; i >= endIdx; i--) {
		ctx.strokeStyle = colors[i % colors.length];
		ctx.fillStyle   = ctx.strokeStyle;
		ctx.lineWidth   = 4;
		ctx.font = settings.font_height + 'px sans-serif';

		var x = dets[i].bbox[0] * scale;
		var y = dets[i].bbox[1] * scale;
		var w = dets[i].bbox[2] * scale;
		var h = dets[i].bbox[3] * scale;

		if (settings.show_mask) {
			var mask = masks[i];
			if (typeof mask == 'undefined') {
				masks[i] = load_RLE(dets[i].mask, hexToRgb(ctx.strokeStyle));
				masks[i].onload = function() { render(); }
			} else {
				ctx.globalAlpha = settings.mask_alpha / 255;
				ctx.drawImage(mask, 0, 0, mask.width * scale, mask.height * scale);
				ctx.globalAlpha = 1;
			}
		}

		if (settings.show_bbox) {
			ctx.strokeRect(x, y, w, h);
			ctx.stroke();
		}

		var text_array = []
		if (settings.show_class)
			text_array.push(dets[i].category);
		if (settings.show_score)
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
