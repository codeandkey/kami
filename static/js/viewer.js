UPDATE_INTERVAL = 100

var board;
var confidenceChart;
var depthChart;

function init_confidence_chart() {
    var ctx = document.getElementById('confidence-chart').getContext('2d');

    confidenceChart = new Chart(ctx, {
        type: 'bar',
        data: [],
        labels: [],
        options: {
            animation: {
                duration: 0
            }
        }
    });
}

function init_depth_chart() {
    var ctx = document.getElementById('depth-chart').getContext('2d');

    depthChart = new Chart(ctx, {
        type: 'line',
        options: {
            animation: {
                duration: 0
            }
        }
    });
}

function update_confidence_chart(tree) {
    // Sort nodes by N
    tree.sort(function(a, b) {
        return a['n'] < b['n']
    })

    var totaln = 0;
    
    for (var i = 0; i < tree.length; ++i) {
        totaln += tree[i]['n']
    }

    confidenceChart.data = {
        datasets: [{
            label: 'Confidence',
            backgroundColor: 'rgb(99, 255, 132)',
            borderColor: 'rgb(99, 255, 132)',
            data: tree.map(function (nd) { return nd.n / totaln; })
        },{
            label: 'Exploration',
            backgroundColor: 'rgb(255, 132, 255)',
            borderColor: 'rgb(255, 255, 132)',
            data: tree.map(function (nd) { return nd.p * (Math.sqrt(totaln) / (nd.n+ 1)); })
        },{
            label: 'Value',
            backgroundColor: 'rgb(0, 255, 255)',
            borderColor: 'rgb(255, 255, 132)',
            data: tree.map(function (nd) { return nd.q; })
        }],
        labels: tree.map(function (nd) { return nd.action; })
    };

    confidenceChart.update();
}

function update_depth_chart(depth_data) {
    depthChart.data = {
        datasets: [{
            label: 'Depth (root)',
            backgroundColor: 'rgb(99, 255, 132)',
            borderColor: 'rgb(99, 255, 132)',
            data: depth_data
        }],
        labels: [...Array(depth_data.length).keys()]
    };

    depthChart.update();
}

function update_tree_full(tree) {
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
            update_confidence_chart(resp['tree'])
        }
        if ('fen' in resp) {
            update_board(resp['fen']);
        }
        if ('depth' in resp) {
            update_depth_chart(resp['depth']);
        }
    });
}

function update_board(fen) {
    board.position(fen, false);
}

$('document').ready(function() {
    init_confidence_chart();
    init_depth_chart();

    interval = setInterval(update_status, UPDATE_INTERVAL);
    board = Chessboard('game-board', 'start');
});