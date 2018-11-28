import './css/index.css';
import Humanplay from './human_play';
import Trainpipeline from './train';

let train;

function get_params() {
    let size = parseInt(document.getElementById('size').value);
    let n_in_row = parseInt(document.getElementById('n_in_row').value);

    if (size && n_in_row && size >= n_in_row) {
        return [size, n_in_row];
    } else {
        alert('请输入棋盘信息再开始');
        return [];
    }

}

function start_train() {
    let [size, n_in_row] = get_params();

    if (size) {
        let best_model = `indexeddb://best_policy_model_${size}*${size}_${n_in_row}`;

        train = new Trainpipeline({width: size, height: size, n_in_row}, best_model);
        train.run();
    }
}

function stop_train() {
    if (train) {
        train.isStop = true;
    }
}

function use_online_model() {

}

function human_play() {
    let [size, n_in_row] = get_params();
    if (size) {
        let human_play = new Humanplay(size, n_in_row);
        human_play.run();
    }
}

function init_click_event() {
    document.getElementById('start_train').addEventListener('click', start_train);
    document.getElementById('use_online_model').addEventListener('click', use_online_model);
    document.getElementById('human_play').addEventListener('click', human_play);
    document.getElementById('stop_train').addEventListener('click', stop_train);
}

init_click_event();