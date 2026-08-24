"""SVG Icon Library for Country-Year Forecast Studio.

Replaces emojis with clean, responsive vector SVG graphics (Feather/Lucide style).
"""
from __future__ import annotations


def get_svg(name: str, size: int = 18, color: str = "currentColor", extra_class: str = "") -> str:
    """Return an inline SVG string for the requested icon name."""
    cls = f'class="svg-icon {extra_class}"' if extra_class else 'class="svg-icon"'
    style = f'style="vertical-align: text-bottom; width: {size}px; height: {size}px; fill: none; stroke: {color}; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; display: inline-block;"'
    
    icons = {
        "globe": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>',
        "forecast": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
        "compass": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
        "shield": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/></svg>',
        "shield_alert": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>',
        "bar_chart": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
        "brain": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M12 5v13"/></svg>',
        "sliders": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
        "check": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><polyline points="20 6 9 17 4 12"/></svg>',
        "check_circle": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        "alert_circle": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
        "info": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        "search": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
        "refresh": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><path d="M21.5 2v6h-6"/><path d="M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>',
        "cpu": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg>',
        "layers": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "database": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
        "arrow_right": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
        "zap": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        "sparkles": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {cls} {style}><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>',
    }
    return icons.get(name, icons["info"])
