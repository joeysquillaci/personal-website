const galleryRoot = document.querySelector("[data-camper-gallery]");

if (galleryRoot) {
  const formatDate = (value) => {
    const [year, month, day] = value.split("-").map(Number);
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    }).format(new Date(year, month - 1, day));
  };

  const renderGallery = (photos) => {
    const groups = {
      progress: photos.filter((photo) => photo.file.startsWith("progress/")),
      v1: photos.filter((photo) => photo.file.startsWith("v1/")),
    };
    let activeGroup = "progress";
    let activeIndex = 0;

    galleryRoot.innerHTML = `
      <div class="camper-gallery__tabs" role="tablist" aria-label="Camper photo galleries">
        <button class="camper-gallery__tab is-active" type="button" data-gallery-group="progress" role="tab" aria-selected="true">Build progress (${groups.progress.length})</button>
        <button class="camper-gallery__tab" type="button" data-gallery-group="v1" role="tab" aria-selected="false">Finished V1 (${groups.v1.length})</button>
      </div>
      <div class="camper-gallery__stage">
        <button class="camper-gallery__control" type="button" data-gallery-direction="previous" aria-label="Previous photo">←</button>
        <figure class="camper-gallery__figure">
          <img class="camper-gallery__image" alt="">
          <figcaption class="camper-gallery__caption"></figcaption>
        </figure>
        <button class="camper-gallery__control" type="button" data-gallery-direction="next" aria-label="Next photo">→</button>
      </div>
      <div class="camper-gallery__thumbnail-list" aria-label="Photo thumbnails"></div>
    `;

    const image = galleryRoot.querySelector(".camper-gallery__image");
    const caption = galleryRoot.querySelector(".camper-gallery__caption");
    const thumbnails = galleryRoot.querySelector(".camper-gallery__thumbnail-list");
    const tabs = [...galleryRoot.querySelectorAll("[data-gallery-group]")];

    const render = () => {
      const photosForGroup = groups[activeGroup];
      const activePhoto = photosForGroup[activeIndex];
      image.src = `assets/${activePhoto.file}`;
      image.alt = `${activeGroup === "v1" ? "Finished V1 camper" : "Camper build progress"} photo captured ${formatDate(activePhoto.captured)}`;
      caption.textContent = `${formatDate(activePhoto.captured)} · ${activeIndex + 1} of ${photosForGroup.length}`;
      thumbnails.innerHTML = photosForGroup.map((photo, index) => `
        <button class="camper-gallery__thumbnail${index === activeIndex ? " is-active" : ""}" type="button" data-gallery-index="${index}" aria-label="View photo ${index + 1}">
          <img src="assets/${photo.file}" alt="" loading="lazy">
        </button>
      `).join("");
      thumbnails.querySelector(`[data-gallery-index="${activeIndex}"]`)?.scrollIntoView({ block: "nearest", inline: "center" });
    };

    tabs.forEach((tab) => tab.addEventListener("click", () => {
      activeGroup = tab.dataset.galleryGroup;
      activeIndex = 0;
      tabs.forEach((item) => {
        const isActive = item === tab;
        item.classList.toggle("is-active", isActive);
        item.setAttribute("aria-selected", String(isActive));
      });
      render();
    }));

    galleryRoot.addEventListener("click", (event) => {
      const indexButton = event.target.closest("[data-gallery-index]");
      if (indexButton) {
        activeIndex = Number(indexButton.dataset.galleryIndex);
        render();
        return;
      }
      const directionButton = event.target.closest("[data-gallery-direction]");
      if (!directionButton) return;
      const length = groups[activeGroup].length;
      activeIndex = directionButton.dataset.galleryDirection === "next"
        ? (activeIndex + 1) % length
        : (activeIndex - 1 + length) % length;
      render();
    });

    render();
  };

  fetch("assets/data/photo-manifest.json")
    .then((response) => response.ok ? response.json() : Promise.reject(new Error("Unable to load photo manifest.")))
    .then(renderGallery)
    .catch(() => {
      galleryRoot.textContent = "The photo gallery could not be loaded.";
    });
}
