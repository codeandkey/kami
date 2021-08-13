UPDATE_INTERVAL = 100

var board;

function update_tree(tree) {
    $('#tree-body').empty();

    // Sort nodes by N
    tree.sort(function(a, b) {
        return a['n'] < b['n']
    })

    var totaln = 0;
    
    for (var i = 0; i < tree.length; ++i) {
        totaln += tree[i]['n']
    }

    // Map into rows
    var rows = tree.map(function(v) {
        var row = '<tr>';

        if (v['action'].length == 4) {
            v['action'] += ' '
        }

        row += '<th scope="col">' + v['action'] + '</th>';
        row += '<td scope="col">' + (v['n'] * 100 / totaln).toFixed(1) + '%</td>';
        row += '<td scope="col">' + v['q'].toFixed(2) + '</td>';
        row += '<td scope="col">' + (v['p'] * 100).toFixed(1) + '%</td>';
        row += '<td scope="col">' + (v['tn'] * 100 / v['n']).toFixed(1)+ '%</td>';

        row += '</tr>';
        return row;
    });

    for (var i = 0; i < rows.length; ++i) {
        $('#tree-body').append(rows[i]);
    }
}

function update_status() {
    jQuery.ajax('/status').done(function(resp) {
        if ('tree' in resp) {
            update_tree(resp['tree'])
        }

        if ('fen' in resp) {
            update_board(resp['fen']);
        }
    });
}

function update_board(fen) {
    board.position(fen, false);
}

$('document').ready(function() {
    interval = setInterval(update_status, UPDATE_INTERVAL);
    board = Chessboard('game-board', 'start');
});