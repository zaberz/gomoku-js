import EventEmitter from 'eventemitter3';

import * as tf from '@tensorflow/tfjs';

export class UI extends EventEmitter {
    constructor() {
        super();
        this.trainProgressElm = document.getElementById('train_progress');

        this.total_width = 1000;
        this.total_height = 1000;

        let canvas = this.canvas = document.createElement('canvas');
        canvas.width = this.total_width;
        canvas.height = this.total_height;
        this.ctx = canvas.getContext('2d');

        canvas.addEventListener('click', this.tapEventHandler.bind(this));
        canvas.addEventListener('tap', this.tapEventHandler.bind(this));

        let container = document.getElementById('board_container');

        container.appendChild(canvas);
    }

    tapEventHandler(e) {
        let pos = this.getMousePos(this.canvas, e);
        let loc = this.calcPos(pos);
        let move = this.posToMove(loc);
        this.emit('get_move', move);
    }

    getMousePos(canvas, event) {
        let rect = canvas.getBoundingClientRect();

        let x = (event.clientX - rect.left) * (canvas.width / rect.width);
        let y = (event.clientY - rect.top) * (canvas.height / rect.height);

        return [x, y];
    }

    calcPos(pos) {
        let x = Math.round(pos[0] / (this.total_width / (this.width + 1)));
        let y = Math.round(pos[1] / (this.total_height / (this.height + 1)));
        return [x, y];
    }

    posToMove(pos) {
        let [x, y] = pos;
        return (y - 1) * this.width + (x - 1);
    }

    draw(board = {}, player1, player2) {
        let {width, height, states} = board;
        this.width = width;
        this.height = height;

        let ctx = this.ctx;

        ctx.fillStyle = '#ffca29';
        ctx.fillRect(0, 0, this.total_width, this.total_height);

        ctx.strokeStyle = '#000000';
        let i = 0;
        let col_width = this.total_width / (width + 1);
        let row_height = this.total_height / (height + 1);
        ctx.beginPath();
        while (i++ < width) {
            ctx.moveTo(i * col_width, 0);
            ctx.lineTo(i * col_width, this.total_height);
        }
        let j = 0;
        while (j++ < height) {
            ctx.moveTo(0, j * row_height);
            ctx.lineTo(this.total_width, j * row_height);
        }
        ctx.lineWidth = 8;
        ctx.stroke();
        ctx.closePath();
        for (let i in states) {
            let h = Math.floor(i / width) + 1;
            let v = i % width + 1;
            let player = states[i];
            if (player === 1) {
                ctx.fillStyle = '#ffffff';
            } else {
                ctx.fillStyle = '#000000';
            }
            ctx.beginPath();
            ctx.arc(v * col_width, h * row_height, col_width / 2.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.closePath();
        }
    }

    set trainProgress(v) {
        this.trainProgressElm.innerText = v;
    }
}

export default new UI();
