/* TaskAware UI helpers: no animation loops, no WebGL.
   - Persists theme (light/dark), Reduce Motion and
     Simple View toggles.
   - Optional CSS-3D chart tilt (off until requested).
   - Animated counters honour reduced motion.
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

    function motionReduced() {
        if (root.getAttribute("data-motion") === "reduced") {
            return true;
        }
        return window.matchMedia(
            "(prefers-reduced-motion: reduce)"
        ).matches;
    }

    function applyTheme() {
        var mode = stored("ta-theme", "light");
        root.setAttribute("data-theme", mode);
        syncPressed("ta-theme-toggle", mode === "dark");
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

    function animateCounters() {
        var nodes = document.querySelectorAll("[data-count-up]");
        if (!nodes.length) {
            return;
        }
        if (motionReduced()) {
            return;
        }
        nodes.forEach(function (el) {
            var target = parseFloat(el.getAttribute("data-count-up"));
            if (isNaN(target)) {
                return;
            }
            var start = null;
            var duration = 900;
            function frame(now) {
                if (!start) {
                    start = now;
                }
                var ratio = Math.min((now - start) / duration, 1);
                el.textContent = Math.round(target * ratio);
                if (ratio < 1) {
                    window.requestAnimationFrame(frame);
                }
            }
            window.requestAnimationFrame(frame);
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        applyTheme();
        applyMotion();
        applyView();
        animateCounters();

        var themeBtn = document.getElementById("ta-theme-toggle");
        if (themeBtn) {
            themeBtn.addEventListener("click", function () {
                var next =
                    root.getAttribute("data-theme") === "dark"
                        ? "light"
                        : "dark";
                store("ta-theme", next);
                applyTheme();
            });
        }

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
