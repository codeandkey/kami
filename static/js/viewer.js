UPDATE_INTERVAL = 250

var board;
var confidenceChart;
var depthChart;
var scoreChart;

function init_confidence_chart() {
    var ctx = document.getElementById('confidence-chart').getContext('2d');

    confidenceChart = new Chart(ctx, {
        responsive: true,
        maintainAspectRatio: false,
        type: 'bar',
        data: [],
        labels: [],
        options: {
            animation: {
                duration: 0
            },
            scales: {
                yAxis: {
                    min: -1,
                    max: 1
                },
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
            },
            scales: {
                yAxis: {
                    min: 0,
                }
            }
        }
    });
}

function init_score_chart() {
    var ctx = document.getElementById('score-chart').getContext('2d');

    scoreChart = new Chart(ctx, {
        type: 'line',
        options: {
            animation: {
                duration: 0
            },
            scales: {
                yAxis: {
                    min: -1,
                    max: 1
                },
                xAxis: {
                    position: 'center',
                    borderWidth: 4,
                    color: '#333333',
                    ticks: {
                        display: false,
                    }
                }
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
            backgroundColor: '#f92672',
            borderColor: '#f92672',
            data: tree.map(function (nd) { return nd.n / totaln; })
        },{
            label: 'Exploration',
            backgroundColor: '#ae81ff',
            borderColor: '#ae81ff',
            data: tree.map(function (nd) { return nd.p * 3.5 * (Math.sqrt(totaln) / (nd.n+ 1)); })
        },{
            label: 'Value',
            backgroundColor: '#66d9ef',
            borderColor: '#66d9ef',
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
            backgroundColor: '#a1ef44',
            borderColor: '#a1ef44',
            data: depth_data
        }],
        labels: [...Array(depth_data.length).keys()]
    };

    depthChart.update();
}

function update_score_chart(score_data) {
    scoreChart.data = {
        datasets: [{
            label: 'Score (w)',
            backgroundColor: '#f92672',
            borderColor: '#f92672',
            data: score_data
        }],
        labels: [...Array(score_data.length).keys()]
    };

    scoreChart.update();
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
        if ('score' in resp) {
            update_score_chart(resp['score']);
        }
        if ('state' in resp) {
            $('#state').text(resp['state']);
        }
        if ('progress' in resp) {
            $('#search-progress').text(Math.round(Number.parseFloat(resp['progress']) * 100) + '%');
            $('#search-progress').css('width', (resp['progress'] * 100).toString() + '%');
        }
        if ('nps' in resp) {
            // Update search data
            $('#search-data').html(
                'NPS: ' + Math.round(resp.nps) + '<br>' +
                'FEN: ' + resp.fen
            );
        }
    });
}

function update_board(fen) {
    board.position(fen, false);
}

$('document').ready(function() {
    init_confidence_chart();
    init_depth_chart();
    init_score_chart();

    interval = setInterval(update_status, UPDATE_INTERVAL);
    board = Chessboard('game-board', 'start');

    $(window).resize(function() {
        confidenceChart.resize();
        depthChart.resize();
        scoreChart.resize();
    });
});