$(document).ready(function() {
	// Load in det_index and fill the ablation list with the appropriate elements
	$.ajax({
		url: 'detindex',
		dataType: 'text',
		success: function (data) {
			data = data.trim().split('\n');
			for (let i = 0; i < data.length; i++) {
				name = data[i];

				$('#ablation_list').append(
					'<li><a href="viewer.html?ablation=' + name + '&idx=0">' + name + '</a></li>'
				);
			}
		}
	});
});
