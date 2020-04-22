(() => {
    const mkElement = (el, attrs) => {
        const e = document.createElement(el);
        for (const [a, v] of Object.entries(attrs)) {
            e.setAttribute(a, v);
        }
        return e;
    };
    const waitForGlobals = (globals, fn, timeout= 15000) => {
        const later = new Date().getTime() + timeout;
        let waittime = 10;
        const waitfn = () => {
            if (new Date().getTime() > later) {
                return;
            }
            for (const check of globals) {
                if (typeof window[check] === "undefined") {
                    setTimeout(waitfn, waittime=Math.min(waittime*1.5, 250));
                    return;
                }
            }
            fn();
        }
        setTimeout(waitfn, waittime);
    }

    const head = document.head;
    head.appendChild(mkElement('link', {rel:'stylesheet', href: './markdownify.css'}));
    head.appendChild(mkElement('link', {rel:'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css'}));
    head.appendChild(mkElement('link', {rel:'stylesheet', href: 'https://cdn.jsdelivr.net/npm/markdown-it-texmath/css/texmath.min.css'}));
    head.appendChild(mkElement('link', {rel:'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css'}));
    head.appendChild(mkElement('script', {src: 'https://cdn.jsdelivr.net/npm/markdown-it/dist/markdown-it.min.js'}));
    head.appendChild(mkElement('script', {src: 'https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js'}));
    head.appendChild(mkElement('script', {src: 'https://cdn.jsdelivr.net/npm/markdown-it-texmath/texmath.min.js'}));



    waitForGlobals(["markdownit", "texmath", "katex"], () => {
        const tm = texmath.use(katex);
        const md = markdownit().use(tm, { engine: katex,
                                          delimiters:'dollars',
                                          macros:{"\\RR": "\\mathbb{R}"}
                                        });
        for (const src of [...document.querySelectorAll('script[type="text/x-markdown"]')].reverse()) {
            const dest = document.createElement('article');
            dest.innerHTML = md.render(src.innerText);
            src.replaceWith(dest);
        }
        const h1 = document.getElementsByTagName('h1');
        if (h1 && h1.length) {
            document.title = h1[0].innerText;
        }
    })
})();


