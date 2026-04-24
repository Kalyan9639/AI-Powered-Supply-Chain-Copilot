// Simple i18n implementation
const translations = {
    en: {
        title: 'Agentic AI Supply Chain Copilot',
        language_label: 'Language:',
        run_cycle_btn: 'Run Agent Cycle',
        approve_btn: 'Approve',
        review_btn: 'Review',
        discard_btn: 'Discard',
        language_select: 'Language',
    },
    hi: {
        title: 'एजेंटिक एआई सप्लाई चेन कॉपिलॉट',
        language_label: 'भाषा:',
        run_cycle_btn: 'एजेंट चक्र चलाएँ',
        approve_btn: 'स्वीकृत',
        review_btn: 'समीक्षा',
        discard_btn: 'रद्द',
        language_select: 'भाषा',
    },
};

function setLanguage(lang) {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (translations[lang] && translations[lang][key]) {
            el.textContent = translations[lang][key];
        }
    });
    localStorage.setItem('appLang', lang);
}

document.addEventListener('DOMContentLoaded', () => {
    const saved = localStorage.getItem('appLang') || 'en';
    setLanguage(saved);
    const select = document.getElementById('lang-select');
    if (select) {
        select.value = saved;
        select.addEventListener('change', e => setLanguage(e.target.value));
    }
});

export { setLanguage };