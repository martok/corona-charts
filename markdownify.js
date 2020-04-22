(() => {
    const mkElement = (el, attrs = {}, content = null) => {
        const e = document.createElement(el);
        for (const [a, v] of Object.entries(attrs)) {
            e.setAttribute(a, v);
        }
        if (typeof content === "string") {
            e.innerText = content;
        } else if (Array.isArray(content)) {
            for (const c of content) {
                e.appendChild(c);
            }
        } else if (!!content) {
            e.appendChild(content);
        }
        return e;
    };
    const waitForGlobals = (globals, fn, timeout = 15000) => {
        const later = new Date().getTime() + timeout;
        let waittime = 10;
        const waitfn = () => {
            if (new Date().getTime() > later) {
                return;
            }
            for (const check of globals) {
                if (typeof window[check] === "undefined") {
                    setTimeout(waitfn, waittime = Math.min(waittime * 1.5, 250));
                    return;
                }
            }
            fn();
        }
        setTimeout(waitfn, waittime);
    }

    const head = document.head;
    head.appendChild(mkElement('link', {rel: 'stylesheet', href: './markdownify.css'}));
    head.appendChild(mkElement('link', {
        rel: 'stylesheet',
        href: 'https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css'
    }));
    head.appendChild(mkElement('link', {
        rel: 'stylesheet',
        href: 'https://cdn.jsdelivr.net/npm/markdown-it-texmath/css/texmath.min.css'
    }));
    head.appendChild(mkElement('link', {
        rel: 'stylesheet',
        href: 'https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css'
    }));
    head.appendChild(mkElement('script', {src: 'https://cdn.jsdelivr.net/npm/markdown-it/dist/markdown-it.min.js'}));
    head.appendChild(mkElement('script', {src: 'https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js'}));
    head.appendChild(mkElement('script', {src: 'https://cdn.jsdelivr.net/npm/markdown-it-texmath/texmath.min.js'}));

    waitForGlobals(["markdownit", "texmath", "katex"], () => {
        const tm = texmath.use(katex);
        const md = markdownit().use(tm, {
            engine: katex,
            delimiters: 'dollars',
        });
        for (const src of [...document.querySelectorAll('script[type="text/x-markdown"]')].reverse()) {
            const dest = document.createElement('article');
            dest.innerHTML = md.render(src.innerText);
            src.replaceWith(dest);
        }
        // use first heading as page title
        const h1 = document.getElementsByTagName('h1');
        if (h1 && h1.length) {
            document.title = h1[0].innerText;
        }
        // create TOC
        const headings = [...document.querySelectorAll('h1,h2,h3')];
        if (headings.length) {
            const ullv1 = mkElement('ul')
            const toc = mkElement('div', {id: 'toc'}, [
                mkElement('div', {'class': 'toctitle'}, 'Contents'),
                ullv1
            ]);
            document.body.insertBefore(toc, document.body.firstChild);
            let container = ullv1;
            let level = 'h1';
            for (const heading of headings) {
                const lv = heading.tagName.toLowerCase();
                if (lv > level) {
                    const newcont = mkElement('ul');
                    container.lastChild.appendChild(newcont);
                    container = newcont;
                    level = lv;
                } else if (lv < level) {
                    container = container.parentElement.parentElement;
                    level = lv;
                }
                const slug = heading.innerText.toLowerCase().replace(/\W/g, '_');
                heading.setAttribute('id', slug);
                container.appendChild(mkElement('li', {}, [
                    mkElement('a', {href: '#' + slug}, heading.innerText)
                ]));
            }
            // made only one h1?
            if (ullv1.childElementCount === 1) {
                const nextDown = ullv1.firstChild.lastChild;
                ullv1.replaceWith(nextDown);
            }
        }
    })
})();


