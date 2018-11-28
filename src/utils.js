export function zip() {
    let args = [].slice.call(arguments);
    let longest = args.reduce(function (a, b) {
        return a.length > b.length ? a : b;
    }, []);

    return longest.map(function (_, i) {
        return args.map(function (array) {
            return array[i];
        });
    });
}

export class Random {
    static choice(data, p) {
        if (data.length !== p.length) {
            throw new Error('error');
        }

        if (Math.round(p.reduce((a, b) => (a + b))) !== 1) {
            throw new Error('sum probability not equal 1');
        }

        let i = Math.random();
        let index = 0;
        let total = 0;
        for (let v of p) {
            total += v;
            if (total > i) {
                break;
            }
            index++;
        }
        return data[index];

    }
}