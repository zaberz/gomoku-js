const path = require('path');

const HtmlWebpackPlugin = require('html-webpack-plugin');
const CleanWebpackPlugin = require('clean-webpack-plugin');
const webpack = require('webpack');

module.exports = {
    entry: './src/index.js',
    output: {
        filename: 'main.js',
        path: path.resolve(__dirname, 'docs'),
    },
    module: {
        rules: [
            {test: /\.css$/, use: ['style-loader', 'css-loader']},
            {
                test: /\.tsx?$/,
                use: [
                    {
                        loader: 'ts-loader',
                        // exclude: /node_modules/,

                        options: {
                            transpileOnly: true,
                            experimentalWatchApi: true,
                        },
                    },
                ],

            },
        ],
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
    },
    devtool: 'inline-source-map',
    devServer: {
        contentBase: './dist',
        // hot: true,
    },
    plugins: [
        new CleanWebpackPlugin(['dist']),

        new HtmlWebpackPlugin({
            title: '',
            template: './index.html'
        }),
        // new webpack.HotModuleReplacementPlugin(),
    ],
};