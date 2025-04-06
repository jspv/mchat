import { loadResource } from "../../static/utils/resources.js";

export default {
    template: `<div><slot /></div>`,
    async mounted() {
        await this.$nextTick();

        const prefix = window.path_prefix ?? '/_nicegui/';
        const css = this.codehilite_css_url;

        if (!css) {
            console.warn('[smart_code] codehilite_css_url prop is missing or empty');
            return;
        }

        const fullUrl = prefix + css;

        try {
            await loadResource(fullUrl);
        } catch (err) {
            console.error('[smart_code] Failed to load resource:', fullUrl);
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
