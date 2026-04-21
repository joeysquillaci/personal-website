const FEATURE_FLAGS = {
  trades: false,
};

function renderRelevantLinksFooter() {
  const pageShell = document.querySelector(".page-shell");

  if (!pageShell || pageShell.querySelector(".site-footer--links")) {
    return;
  }

  const footer = document.createElement("footer");
  footer.className = "site-footer site-footer--links";
  footer.innerHTML = `
    <div class="container site-footer__inner">
      <p class="site-footer__title">Relevant Links</p>
      <div class="site-footer__links" aria-label="Relevant links">
        <a
          class="site-footer__link magnetic"
          href="https://www.linkedin.com/in/joey-squillaci-4795891a3/"
          target="_blank"
          rel="noreferrer"
          aria-label="LinkedIn profile"
          title="LinkedIn"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M5.4 3.5A1.9 1.9 0 1 1 5.4 7.3a1.9 1.9 0 0 1 0-3.8Zm-1.7 5h3.3V20H3.7V8.5Zm5.4 0h3.2V10h.1c.4-.8 1.6-1.8 3.3-1.8 3.5 0 4.2 2.3 4.2 5.3V20h-3.3v-5.8c0-1.4 0-3.2-2-3.2s-2.2 1.5-2.2 3.1V20H9.1V8.5Z"></path>
          </svg>
        </a>
        <a
          class="site-footer__link magnetic"
          href="https://github.com/joeysquillaci"
          target="_blank"
          rel="noreferrer"
          aria-label="GitHub profile"
          title="GitHub"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M12 .6a11.4 11.4 0 0 0-3.6 22.2c.6.1.8-.3.8-.6v-2c-3.2.7-3.9-1.3-3.9-1.3-.5-1.4-1.3-1.7-1.3-1.7-1.1-.8.1-.8.1-.8 1.2.1 1.9 1.3 1.9 1.3 1 .1.8 2.8 3.2 2 .1-.8.4-1.3.7-1.6-2.5-.3-5.1-1.3-5.1-5.8 0-1.3.5-2.4 1.3-3.2-.2-.3-.6-1.5.1-3.1 0 0 1.1-.4 3.4 1.2a11.2 11.2 0 0 1 6.1 0c2.3-1.6 3.4-1.2 3.4-1.2.7 1.6.3 2.8.1 3.1.8.8 1.3 1.9 1.3 3.2 0 4.5-2.7 5.5-5.2 5.8.4.4.8 1.1.8 2.2v3.2c0 .3.2.7.8.6A11.4 11.4 0 0 0 12 .6Z"></path>
          </svg>
        </a>
      </div>
    </div>
  `;

  pageShell.appendChild(footer);
}

function isFeatureEnabled(featureName) {
  if (!featureName) {
    return true;
  }

  return Boolean(FEATURE_FLAGS[featureName]);
}

function applyFeatureVisibility() {
  const featureElements = document.querySelectorAll("[data-feature]");

  featureElements.forEach((element) => {
    const featureName = element.dataset.feature;
    const isVisible = isFeatureEnabled(featureName);
    element.classList.toggle("hidden-content", !isVisible);
  });
}

const projects = [
  {
    id: "carry-select-adder",
    slug: "16-bit-carry-select-adder",
    category: "VLSI Design",
    title: "16-Bit Carry Select Adder",
    summary:
      "Designed and verified a custom 16-bit carry-select adder in Cadence Virtuoso, achieving a 410.4 ps schematic delay and a 1.245 ns post-layout delay while staying within project constraints.",
    tags: ["Cadence Virtuoso", "CMOS Design", "Timing Analysis"],
    pageHref: "projects/16-bit-carry-select-adder/index.html",
  },
  {
    id: "military-bcds",
    slug: "inventus-internship",
    category: "Embedded Systems",
    title: "Inventus Internship",
    summary:
      "High-level case study of internship work involving embedded software, serial communication, battery-system testing, documentation, and production-floor engineering support.",
    tags: ["Python", "Embedded C", "Battery Systems"],
    pageHref: "projects/inventus-internship/index.html",
  },
  {
    id: "cad-viewer-placeholder",
    slug: "ornament-project",
    category: "Personal Build",
    title: "ESP32 Ornament",
    summary:
      "Interactive holiday ornament built around an ESP32, charging module, buck-boost converter, and e-ink display to show rotating Christmas-themed messages.",
    tags: ["ESP32", "E-Ink Display", "3D Printing"],
    pageHref: "projects/ornament-project/index.html",
  },
  {
    id: "equity-education",
    slug: "equity-education",
    category: "Education Platform",
    title: "Equity Education",
    summary:
      "Educational equity analysis prototype using an LSTM model and interactive GUI to connect predictions with technical market context.",
    tags: ["LSTM", "Tkinter GUI", "Equity Analysis"],
    pageHref: "projects/equity-education/index.html",
  },
  {
    id: "trades-ai-training",
    feature: "trades",
    slug: "trades-ai-training",
    category: "AI Training",
    title: "TRADES: AI Training",
    summary:
      "Tiny reproduction of TRADES on CIFAR-10 with smaller-batch tuning, parameter sweeps, and PGD-based robustness evaluation.",
    tags: ["TRADES", "CIFAR-10", "PGD Attack"],
    pageHref: "projects/trades-ai-training/index.html",
  },
];

const experience = [
  {
    id: "international-motors",
    role: "Lead System Engineer",
    company: "International Motors",
    period: "January 2024 - Present",
    summary:
      "Working on vehicle network integration and diagnostics with a focus on SAE and J1939 standards, signal definition, DBC maintenance, and communication analysis across module-based systems.",
    highlights: [
      "CAN / J1939 network implementation and analysis",
      "DBC creation and signal/message definition",
      "Cross-functional supplier and standards coordination",
    ],
  },
  {
    id: "inventus-power",
    role: "Software/Electrical Engineering Intern",
    company: "Inventus Power",
    period: "May 2023 - August 2023",
    summary:
      "Contributed to embedded and electrical engineering workflows involving battery management systems, command-line tooling, serial communication, hardware interfacing, and test-focused software development.",
    highlights: [
      "Python and embedded C application development",
      "Serial communication over USB, RS232, UART, CAN, and I2C",
      "Hardware troubleshooting and live sensor data collection",
    ],
  },
  {
    id: "apple",
    role: "Product Specialist",
    company: "Apple",
    period: "August 2019 - January 2024",
    summary:
      "Delivered customer-facing technical support across Apple hardware and software, combining troubleshooting, communication, and problem-solving in a high-volume retail environment.",
    highlights: [
      "Technical troubleshooting across Apple devices",
      "Customer communication and issue resolution",
      "Peer support and team resource responsibilities",
    ],
  },
];

function renderProjects() {
  const projectsGrid = document.querySelector("#projects-grid");

  if (!projectsGrid) {
    return;
  }

  projectsGrid.innerHTML = projects
    .filter((project) => isFeatureEnabled(project.feature))
    .map((project) => {
      const tags = project.tags
        .map((tag) => `<span>${tag}</span>`)
        .join("");
      const isClickable = Boolean(project.pageHref);
      const resourceLink = project.resourceHref
        ? `<a class="project-card__resource" href="${project.resourceHref}" target="_blank" rel="noreferrer">${project.resourceLabel || "View Resource"}</a>`
        : "";

      return `
        <article
          class="project-card reveal tilt-card${isClickable ? " project-card--clickable" : ""}"
          data-project-id="${project.id}"
          data-project-slug="${project.slug}"
          ${project.pageHref ? `data-project-url="${project.pageHref}" tabindex="0" role="link"` : ""}
        >
          <h3>${project.title}</h3>
          <p>${project.summary}</p>
          <div class="project-card__tags">
            ${tags}
          </div>
          ${resourceLink}
        </article>
      `;
    })
    .join("");
}

function renderExperience() {
  const experienceGrid = document.querySelector("#experience-grid");

  if (!experienceGrid) {
    return;
  }

  experienceGrid.innerHTML = experience
    .map((item) => {
      const highlights = item.highlights
        .map((highlight) => `<li>${highlight}</li>`)
        .join("");

      return `
        <article class="experience-card reveal">
          <h3 class="experience-card__role">${item.role}</h3>
          <p class="experience-card__company">${item.company}</p>
          <p class="experience-card__period">${item.period}</p>
          <p>${item.summary}</p>
          <ul class="experience-card__list">
            ${highlights}
          </ul>
        </article>
      `;
    })
    .join("");
}

renderProjects();
renderExperience();
applyFeatureVisibility();
renderRelevantLinksFooter();

const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector(".nav-links");
const navAnchors = [...document.querySelectorAll(".nav-links a:not(.hidden-content)")];
const navGroupToggle = document.querySelector(".nav-group__toggle");
const navGroup = document.querySelector(".nav-group");
const body = document.body;
const revealItems = document.querySelectorAll(".reveal");
const counters = document.querySelectorAll(".counter");
const tiltCards = document.querySelectorAll(".tilt-card");
const magneticButtons = document.querySelectorAll(".magnetic");
const sections = [...document.querySelectorAll("main section[id]")];
const projectCards = document.querySelectorAll("[data-project-url]");
const root = document.documentElement;

if (navToggle && navLinks) {
  function setProjectsDropdownOpen(isOpen) {
    if (!navGroup || !navGroupToggle) {
      return;
    }

    navGroup.classList.toggle("is-open", isOpen);
    navGroupToggle.setAttribute("aria-expanded", String(isOpen));
  }

  function setNavOpenState(isOpen) {
    navLinks.classList.toggle("is-open", isOpen);
    navToggle.setAttribute("aria-expanded", String(isOpen));
    body.classList.toggle("nav-drawer-open", isOpen);

    if (!isOpen) {
      setProjectsDropdownOpen(false);
    }
  }

  navToggle.addEventListener("click", () => {
    const isOpen = !navLinks.classList.contains("is-open");
    setNavOpenState(isOpen);
  });

  if (navGroup && navGroupToggle) {
    navGroupToggle.addEventListener("click", (event) => {
      event.stopPropagation();
      const isOpen = !navGroup.classList.contains("is-open");
      setProjectsDropdownOpen(isOpen);
    });
  }

  navAnchors.forEach((anchor) => {
    anchor.addEventListener("click", () => {
      // Let mobile browsers process navigation first, then close the drawer.
      window.setTimeout(() => {
        setNavOpenState(false);
      }, 0);
    });
  });

  document.addEventListener("click", (event) => {
    if (
      navLinks.classList.contains("is-open") &&
      !navLinks.contains(event.target) &&
      !navToggle.contains(event.target)
    ) {
      setNavOpenState(false);
      return;
    }

    if (navGroup && !navGroup.contains(event.target)) {
      setProjectsDropdownOpen(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (navLinks.classList.contains("is-open")) {
        setNavOpenState(false);
      } else {
        setProjectsDropdownOpen(false);
      }
    }
  });
}

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  {
    threshold: 0.2,
  }
);

revealItems.forEach((item, index) => {
  item.style.transitionDelay = `${index % 3 === 0 ? 0 : (index % 3) * 120}ms`;
  revealObserver.observe(item);
});

function animateCounter(counter) {
  const target = Number(counter.dataset.target || 0);
  const duration = 1400;
  const startTime = performance.now();

  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    counter.textContent = Math.round(target * eased).toString();

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

const counterObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        counterObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.8 }
);

counters.forEach((counter) => counterObserver.observe(counter));

window.addEventListener("pointermove", (event) => {
  const x = `${(event.clientX / window.innerWidth) * 100}%`;
  const y = `${(event.clientY / window.innerHeight) * 100}%`;
  root.style.setProperty("--pointer-x", x);
  root.style.setProperty("--pointer-y", y);
});

tiltCards.forEach((card) => {
  card.addEventListener("pointermove", (event) => {
    const bounds = card.getBoundingClientRect();
    const px = (event.clientX - bounds.left) / bounds.width;
    const py = (event.clientY - bounds.top) / bounds.height;
    const rotateY = (px - 0.5) * 10;
    const rotateX = (0.5 - py) * 10;

    card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
  });

  card.addEventListener("pointerleave", () => {
    card.style.transform = "";
  });
});

magneticButtons.forEach((button) => {
  button.addEventListener("pointermove", (event) => {
    const bounds = button.getBoundingClientRect();
    const offsetX = event.clientX - (bounds.left + bounds.width / 2);
    const offsetY = event.clientY - (bounds.top + bounds.height / 2);

    button.style.transform = `translate(${offsetX * 0.12}px, ${offsetY * 0.12}px)`;
  });

  button.addEventListener("pointerleave", () => {
    button.style.transform = "";
  });
});

projectCards.forEach((card) => {
  const targetUrl = card.dataset.projectUrl;

  if (!targetUrl) {
    return;
  }

  card.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      return;
    }

    window.location.href = targetUrl;
  });

  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      window.location.href = targetUrl;
    }
  });
});

const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      const link = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
      if (!link) {
        return;
      }

      if (entry.isIntersecting) {
        navAnchors.forEach((anchor) => anchor.classList.remove("is-active"));
        link.classList.add("is-active");
      }
    });
  },
  {
    rootMargin: "-40% 0px -45% 0px",
    threshold: 0,
  }
);

sections.forEach((section) => sectionObserver.observe(section));
