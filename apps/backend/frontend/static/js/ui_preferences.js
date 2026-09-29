(function () {
    const KEY_LANG = 'lang';
    const KEY_THEME = 'theme';
    const KEY_MENU = 'menuCollapsed';

    const validLangs = new Set(['fa', 'en']);
    const validThemes = new Set(['dark', 'light']);

    function getLang() {
        const stored = localStorage.getItem(KEY_LANG);
        return validLangs.has(stored) ? stored : 'fa';
    }

    function getTheme() {
        const stored = localStorage.getItem(KEY_THEME);
        return validThemes.has(stored) ? stored : 'dark';
    }

    function isMenuCollapsed() {
        return localStorage.getItem(KEY_MENU) === '1';
    }

    function applyRootState() {
        const lang = getLang();
        const theme = getTheme();
        const html = document.documentElement;
        html.lang = lang;
        html.dir = lang === 'fa' ? 'rtl' : 'ltr';

        if (document.body) {
            document.body.classList.remove('dark', 'light');
            document.body.classList.add(theme);
        }
    }

    function applyMenuState() {
        const collapsed = isMenuCollapsed();
        const sidePanel = document.getElementById('sidePanel');
        if (document.body) {
            document.body.classList.toggle('menu-collapsed', collapsed);
        }
        if (sidePanel) {
            sidePanel.classList.toggle('collapsed', collapsed);
        }
    }

    function setLang(lang) {
        const next = validLangs.has(lang) ? lang : 'fa';
        localStorage.setItem(KEY_LANG, next);
        applyRootState();
        return next;
    }

    function setTheme(theme) {
        const next = validThemes.has(theme) ? theme : 'dark';
        localStorage.setItem(KEY_THEME, next);
        applyRootState();
        return next;
    }

    function setMenuCollapsed(collapsed) {
        localStorage.setItem(KEY_MENU, collapsed ? '1' : '0');
        applyMenuState();
        return collapsed;
    }

    function toggleMenuCollapsed() {
        return setMenuCollapsed(!isMenuCollapsed());
    }

    window.JBDUI = {
        getState: function () {
            return {
                lang: getLang(),
                theme: getTheme(),
                menuCollapsed: isMenuCollapsed()
            };
        },
        setLang: setLang,
        setTheme: setTheme,
        setMenuCollapsed: setMenuCollapsed,
        toggleMenuCollapsed: toggleMenuCollapsed,
        applyRootState: applyRootState,
        applyMenuState: applyMenuState
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            applyRootState();
            applyMenuState();
        });
    } else {
        applyRootState();
        applyMenuState();
    }

    window.addEventListener('storage', function (event) {
        if (!event.key || event.key === KEY_LANG || event.key === KEY_THEME) {
            applyRootState();
        }
        if (!event.key || event.key === KEY_MENU) {
            applyMenuState();
        }
    });
})();
