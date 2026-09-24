/* TaskAware UI helpers: no animation loops, no WebGL.
   - Persists Reduce Motion / Simple View toggles.
   - Optional CSS-3D chart tilt (off until requested).
   - Timeline uses native semantics; nothing decorative
     delays content. */
(function () {
    "use strict";

    var root = document.documentElement;

    function stored(key, fallback) {
        try {
            return window.localStorage.getItem(key) || fallback;
        } catch (e) {
            return fallback;
        }
    }

    function store(key, value) {
        try {
            window.localStorage.setItem(key, value);
        } catch (e) {
            /* storage unavailable: session-only */
        }
    }

    function applyMotion() {
        var mq = window.matchMedia("(prefers-reduced-motion: reduce)");
        var mode = stored("ta-motion", mq.matches ? "reduced" : "full");
        root.setAttribute("data-motion", mode);
        syncPressed("ta-motion-toggle", mode === "reduced");
    }

    function applyView() {
        var mode = stored("ta-view", "rich");
        root.setAttribute("data-view", mode);
        syncPressed("ta-simple-toggle", mode === "simple");
    }

    function syncPressed(id, pressed) {
        var el = document.getElementById(id);
        if (el) {
            el.setAttribute("aria-pressed", pressed ? "true" : "false");
        }
    }

    document.addEventListener("DOMContentLoaded", function () {
        applyMotion();
        applyView();

        var motionBtn = document.getElementById("ta-motion-toggle");
        if (motionBtn) {
            motionBtn.addEventListener("click", function () {
                var next =
                    root.getAttribute("data-motion") === "reduced"
                        ? "full"
                        : "reduced";
                store("ta-motion", next);
                applyMotion();
            });
        }

        var simpleBtn = document.getElementById("ta-simple-toggle");
        if (simpleBtn) {
            simpleBtn.addEventListener("click", function () {
                var next =
                    root.getAttribute("data-view") === "simple"
                        ? "rich"
                        : "simple";
                store("ta-view", next);
                applyView();
            });
        }

        /* Optional CSS-3D tilt for the 2D cluster chart.
           Static image + data table remain the default. */
        var tiltBtn = document.getElementById("ta-tilt-toggle");
        var stage = document.getElementById("ta-chart-stage");
        if (tiltBtn && stage) {
            tiltBtn.addEventListener("click", function () {
                var on = stage.classList.toggle("ta-chart-3d");
                tiltBtn.setAttribute(
                    "aria-pressed", on ? "true" : "false"
                );
            });
        }
    });
})();
