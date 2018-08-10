$(document).ready(function() {
	// Load in det_index and fill the config list with the appropriate elements
	$.ajax({
		url: 'detindex',
		dataType: 'text',
		success: function (data) {
			data = data.trim().split('\n');
			for (let i = 0; i < data.length; i++) {
				name = data[i];

				$('#config_list').append(
					'<li><a href="viewer.html?config=' + name + '&idx=0">' + name + '</a></li>'
				);
			}
		}
	});
});
