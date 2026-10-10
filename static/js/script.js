(() => {
  const toggle = document.querySelector('[data-menu-toggle]');
  const nav = document.querySelector('[data-nav]');
  if (toggle && nav) {
    toggle.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
    });
    nav.querySelectorAll('a').forEach((link) => {
      if (new URL(link.href, window.location.href).pathname === window.location.pathname) {
        link.setAttribute('aria-current', 'page');
      }
      link.addEventListener('click', () => {
        nav.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  const motionAllowed = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const pointerTiltAllowed = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  if (motionAllowed && pointerTiltAllowed) {
    document.querySelectorAll('[data-tilt], .stat-card, .hero-stat').forEach((card) => {
      card.addEventListener('pointermove', (event) => {
        const rect = card.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width;
        const y = (event.clientY - rect.top) / rect.height;
        card.style.setProperty('--tilt-x', `${(0.5 - y) * 3}deg`);
        card.style.setProperty('--tilt-y', `${(x - 0.5) * 3}deg`);
        card.classList.add('tilt-active');
      });
      card.addEventListener('pointerleave', () => {
        card.style.removeProperty('--tilt-x');
        card.style.removeProperty('--tilt-y');
        card.classList.remove('tilt-active');
      });
    });

    const gameWorld = document.querySelector('.game-world');
    if (gameWorld) {
      const sceneStyle = document.documentElement.style;
      window.addEventListener('pointermove', (event) => {
        const shiftX = (window.innerWidth / 2 - event.clientX) / window.innerWidth * 12;
        const shiftY = (window.innerHeight / 2 - event.clientY) / window.innerHeight * 8;
        sceneStyle.setProperty('--scene-x', `${shiftX.toFixed(1)}px`);
        sceneStyle.setProperty('--scene-y', `${shiftY.toFixed(1)}px`);
      }, { passive: true });
      document.documentElement.addEventListener('pointerleave', () => {
        sceneStyle.removeProperty('--scene-x');
        sceneStyle.removeProperty('--scene-y');
      });
    }
  }

  document.querySelectorAll('[data-probability]').forEach((element) => {
    const probability = Number(element.dataset.probability);
    if (Number.isFinite(probability)) {
      element.style.setProperty('--probability', `${Math.max(0, Math.min(100, probability))}%`);
    }
  });

  document.querySelectorAll('[data-win-rate]').forEach((input) => {
    input.addEventListener('input', () => {
      const value = Number(input.value);
      input.setCustomValidity(Number.isFinite(value) && value >= 0 && value <= 1 ? '' : 'Enter a value from 0 to 1.');
    });
  });
})();
