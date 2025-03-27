import { loadResource } from "../../static/utils/resources.js";

export default {
    template: `<div><slot /></div>`,
    async mounted() {
        await this.$nextTick();

        const prefix = window.path_prefix ?? '/_nicegui/';
        const css = this.codehilite_css_url;

        console.group('[NiceGUI] <NewCode> Resource Load Debug');

        if (!css) {
            console.warn('[NewCode] codehilite_css_url prop is missing or empty');
            console.groupEnd();
            return;
        }

        const fullUrl = prefix + css;
        console.log('[NewCode] Attempting to load:', fullUrl);

        try {
            await loadResource(fullUrl);
            console.log('[NewCode] Resource loaded successfully:', fullUrl);
        } catch (err) {
            console.error('[NewCode] Failed to load resource:', fullUrl);
            console.error(err);
        }

        console.groupEnd();
    },
    props: {
        codehilite_css_url: {
            type: String,
            required: true,
        },
    },
};
