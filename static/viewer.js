UPDATE_INTERVAL = 100

function update_tree(tree) {
    $('#tree-body').empty();

    var next_row = '<tr>';

    next_row += '<td scope="col">' + resp['']

    $('#tree-body').append('<tr>')
}

function update_status() {
    jQuery.ajax('/status').done(function(resp) {
        console.log(resp);
    });
}

interval = setInterval(update_status, UPDATE_INTERVAL);