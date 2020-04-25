/*
    Markdownify.js - (c) 2020 Martok, released under MIT License
    Process any <script type="text/x-markdown"> in the document to rendered content, using markdown-it.

    Features:
        - Markdown-it default https://markdown-it.github.io/
        - heading link slug generation
        - TOC generation
        - on supported browsers: texmath with $delimiters$

    Browser support:
        Bare markdown-it is ES5 + very little extra, so we try to match that for baseline features.
         - use var instead of let
         - closure scopes instead of block scopes
         - polyfills for object iteration
         - textContent instead of innerText (this is semantically problematic)
         - spread function call is not supported, but spread array construction is (?!)
 */

(() => {
    "use strict"
    if (!Object.entries) {
        Object.entries = function( obj ){
            var ownProps = Object.keys( obj ),
                i = ownProps.length,
            resArray = new Array(i); // preallocate the Array
            while (i--)
                resArray[i] = [ownProps[i], obj[ownProps[i]]];
            return resArray;
        };
    }
    if (!Object.values) {
        Object.values = function( obj ){
            var ownProps = Object.keys( obj ),
                i = ownProps.length,
            resArray = new Array(i); // preallocate the Array
            while (i--)
                resArray[i] = obj[ownProps[i]];
            return resArray;
        };
    }

    const mkElement = (el, attrs = {}, content = null) => {
        const e = document.createElement(el);
        for (var [a, v] of Object.entries(attrs)) {
            e.setAttribute(a, v);
        }
        if (typeof content === "string") {
            e.textContent = content;
        } else if (Array.isArray(content)) {
            for (var c of content) {
                e.appendChild(c);
            }
        } else if (!!content) {
            e.appendChild(content);
        }
        return e;
    };
    const waitForScripts = (scripts, fn, timeout = 15000) => {
        const later = new Date().getTime() + timeout;
        var waittime = 10;
        var loading = {};
        const waitfn = () => {
            if (new Date().getTime() > later) {
                return;
            }
            if (Object.values(loading).some(r => typeof r !== "boolean")) {
                setTimeout(waitfn, waittime = Math.min(waittime * 1.5, 250));
                return;
            }
            fn.apply(this, Object.keys(loading).map(k => window[k]));
        }
        for (var n of Object.keys(scripts)) {
            loading[n] = ((name, url) => {
                const scr = mkElement('script');
                scr.onload = () => {loading[name] = true};
                scr.onerror = () => {loading[name] = false};
                scr.src = url;
                document.head.appendChild(scr);
                return scr;
            })(n, scripts[n]);
        }
        setTimeout(waitfn, waittime);
    }

    const head = document.head;
    head.appendChild(mkElement('link', {rel: 'stylesheet', href: './markdownify.css'}));
    head.appendChild(mkElement('link', {rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css'}));
    head.appendChild(mkElement('link', {rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/markdown-it-texmath/css/texmath.min.css'}));
    head.appendChild(mkElement('link', {rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css'}));
    const scripts = {
        'markdownit': 'https://cdn.jsdelivr.net/npm/markdown-it/dist/markdown-it.min.js',
        'katex': 'https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js',
        'texmath': 'https://cdn.jsdelivr.net/npm/markdown-it-texmath/texmath.min.js',
    }

    waitForScripts(scripts, (markdownit, katex, texmath) => {
        const md = markdownit();
        if (katex && texmath) {
            const tm = texmath.use(katex);
            md.use(tm, {
                engine: katex,
                delimiters: 'dollars',
            });
        }
        for (var src of [...document.querySelectorAll('script[type="text/x-markdown"]')].reverse()) {
            const dest = document.createElement('article');
            dest.innerHTML = md.render(src.text);
            src.parentNode.replaceChild(dest, src);
        }
        // use first heading as page title
        const h1 = document.getElementsByTagName('h1');
        if (h1 && h1.length) {
            document.title = h1[0].textContent;
        }
        // create TOC
        const headings = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6,h7,h8,h9')];
        if (headings.length) {
            const ullv1 = mkElement('ul')
            const toc = mkElement('div', {id: 'toc'}, [
                mkElement('div', {'class': 'toctitle'}, 'Contents'),
                ullv1
            ]);
            document.body.insertBefore(toc, document.body.firstChild);
            var container = ullv1;
            var level = 'h1';
            for (var heading of headings) {
                const lv = heading.tagName.toLowerCase();
                const htext = heading.textContent;
                const slug = htext.toLowerCase().replace(/\W/g, '_');
                heading.setAttribute('id', slug);
                if (lv > 'h3')
                    continue;
                if (lv > level) {
                    const newcont = mkElement('ul');
                    container.lastChild.appendChild(newcont);
                    container = newcont;
                    level = lv;
                } else if (lv < level) {
                    container = container.parentElement.parentElement;
                    level = lv;
                }
                container.appendChild(mkElement('li', {}, [
                    mkElement('a', {href: '#' + slug}, htext)
                ]));
            }
            // made only one h1?
            if (ullv1.childElementCount === 1) {
                const nextDown = ullv1.firstChild.lastChild;
                ullv1.parentNode.replaceChild(nextDown, ullv1);
            }
        }
    })
})();


