# services/alerts.py
import streamlit as st

def play_beep(duration_ms: int = 500, freq_hz: int = 880):
    st.components.v1.html(f"""
    <script>
      (function(){{
        try {{
          const AC = window.AudioContext || window.webkitAudioContext;
          const ctx = new AC();
          const o = ctx.createOscillator();
          const g = ctx.createGain();
          o.type = "sine";
          o.frequency.value = {freq_hz};
          o.connect(g); g.connect(ctx.destination);
          g.gain.setValueAtTime(0.001, ctx.currentTime);
          g.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.02);
          o.start();
          g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + {duration_ms}/1000.0);
          o.stop(ctx.currentTime + {duration_ms}/1000.0 + 0.05);
        }} catch(e) {{ console.log("beep error", e); }}
      }})();
    </script>
    """, height=0)

def alert_banner(msg: str):
    st.markdown(
        "<div style='padding:12px 16px;border-radius:10px;border:2px solid #ffb3b3;background:#fff1f1;color:#b00020;font-weight:700;font-size:20px;'>"
        f"🚨 {msg}"
        "</div>",
        unsafe_allow_html=True
    )
    play_beep(380, 920); play_beep(380, 720)