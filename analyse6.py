import pandas as pd                                    # Tabellenwerkzeug
import numpy as np                                     # Rechenwerkzeug für Zahlenreihen
from sklearn.ensemble import RandomForestRegressor     # Random Forest, das Vorhersagemodell
from sklearn.metrics import mean_absolute_error        # Fehlermaß (MAE)
import shap                                            # zur Berechnung der Shapley-Werte
import matplotlib.pyplot as plt                        # Grafikwerkzeug
import matplotlib.dates as mdates                      # für die Datumsbeschriftung der Achse


ORDNER = "daten/"                   # Unterordner mit den CSV-Dateien
KRIEG = pd.Timestamp("2026-02-28")  # Datum des Kriegsausbruchs
W0 = pd.Timestamp("2026-01-15")     # Anfang des Grafik-Zeitfensters
W1 = pd.Timestamp("2026-05-15")     # Ende des Grafik-Zeitfensters


def lade_cds(datei):                                   # liest eine CDS-Datei ein und bereinigt sie
    df = pd.read_csv(ORDNER + datei)                   # CSV einlesen
    df.columns = [c.strip().strip('"').replace("\ufeff", "") for c in df.columns]   # Spaltennamen säubern
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")   # Datum im US-Format (Monat/Tag/Jahr) lesen
    df["cds"] = (df["Price"].astype(str)               # Preis als Text behandeln
                 .str.replace('"', "")                 # Anführungszeichen entfernen
                 .str.replace(",", "")                 # Tausender-Kommas entfernen
                 .astype(float))                       # in eine Zahl umwandeln
    return df[["Date", "cds"]].sort_values("Date").reset_index(drop=True)   # nach Datum sortiert zurückgeben


def lade_reihe(datei, name):                           # liest eine GDELT-Reihe ein
    df = pd.read_csv(ORDNER + datei)                   # CSV einlesen
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]   # Spaltennamen säubern
    df["Date"] = pd.to_datetime(df["Date"])            # Datum umwandeln
    df = df.rename(columns={"Value": name})            # Spalte "Value" in den Wunschnamen umbenennen
    df = ausreisser_entfernen(df, name)                # fehlerhafte Werte herausfiltern
    return df[["Date", name]].sort_values("Date").reset_index(drop=True)   # nach Datum sortiert zurückgeben


def lade_fred(datei, spalte, name):                    # liest eine Finanzreihe (VIX/US-Zins) ein
    df = pd.read_csv(ORDNER + datei)                   # CSV einlesen
    df.columns = [c.strip() for c in df.columns]       # Spaltennamen säubern
    df["Date"] = pd.to_datetime(df["observation_date"])   # Datumsspalte umbenennen und umwandeln
    df[name] = pd.to_numeric(df[spalte], errors="coerce")   # Werte in Zahlen umwandeln, Lücken werden leer
    return df[["Date", name]].dropna().sort_values("Date").reset_index(drop=True)   # leere Tage entfernen, sortieren


def ausreisser_entfernen(df, name):                    # filtert unmögliche Werte heraus
    if "tone" in name.lower():                         # Tonfall: plausibel zwischen -20 und +20
        df.loc[(df[name] < -20) | (df[name] > 20), name] = np.nan
    else:                                              # Volumen: plausibel zwischen 0 und 10
        df.loc[(df[name] < 0) | (df[name] > 10), name] = np.nan
    df[name] = df[name].ffill()                        # Lücken mit dem Vortageswert füllen
    return df


vix = lade_fred("CBOE_Volatility_Index_VIX.csv", "VIXCLS", "VIX")   # VIX einlesen
fed = lade_fred("Market_Yield_on_U_S__Treasury_Securities_at_2-Year_Constant_Maturity__Quoted_on_an_Investment_Basis.csv", "DGS2", "UST2Y")   # zweijährige US-Rendite einlesen


laender = {                                            # ordnet jedem Land seine acht Dateien zu
    "Israel": ("Israel_CDS_Kopie.csv", "GPR_VOL_ISRAEL_Kopie.csv", "GPR_TONE_ISRAEL_Kopie.csv", "EPU_VOL_ISRAEL_Kopie.csv",
               "ECO_VOL_ISRAEL.csv", "ECO_TONE_ISRAEL.csv", "INT_VOL_ISRAEL.csv", "INT_TONE_ISRAEL.csv"),
    "Saudi-Arabien": ("Saudi_Arabien_CDS_Kopie.csv", "GPR_VOL_SAUDI_ARABIEN_Kopie.csv", "GPR_TONE_SAUDI_ARABIEN_Kopie.csv", "EPU_VOL_SAUDI_ARABIEN_Kopie.csv",
               "ECO_VOL_SAUDI_ARABIEN.csv", "ECO_TONE_SAUDI_ARABIEN.csv", "INT_VOL_SAUDI_ARABIEN.csv", "INT_TONE_SAUDI_ARABIEN.csv"),
}


def glaetten_standardisieren(serie):                   # glättet eine Reihe und macht sie vergleichbar
    serie = serie.rolling(28, min_periods=1).mean()    # 28-Tage-Durchschnitt gegen tägliches Rauschen
    return (serie - serie.mean()) / serie.std()        # Z-Standardisierung für die Vergleichbarkeit


treiber = ["GPR", "EPU", "ECO", "INT", "VIX", "UST2Y"]   # die sechs Treiber
ergebnisse = {}                                        # speichert die Ergebnisse je Land


for land, fs in laender.items():                       # Schleife über die Länder
    cf, gv, gt, ev, ecv, ect, inv, intt = fs           # Dateinamen auf einzelne Variablen verteilen
    d = (lade_cds(cf)                                  # CDS-Tabelle öffnen
         .merge(lade_reihe(gv, "GPR_vol"), on="Date", how="left")
         .merge(lade_reihe(gt, "GPR_tone"), on="Date", how="left")
         .merge(lade_reihe(ev, "EPU_vol"), on="Date", how="left")
         .merge(lade_reihe(ecv, "ECO_vol"), on="Date", how="left")   # alle Reihen zu einer Tabelle zusammenführen
         .merge(lade_reihe(ect, "ECO_tone"), on="Date", how="left")
         .merge(lade_reihe(inv, "INT_vol"), on="Date", how="left")
         .merge(lade_reihe(intt, "INT_tone"), on="Date", how="left")
         .merge(vix, on="Date", how="left")
         .merge(fed, on="Date", how="left")
         .sort_values("Date").reset_index(drop=True))  # nach Datum sortieren

    d["GPR"] = -(d["GPR_tone"]) * d["GPR_vol"]          # Stimmungsindikatoren: Ton mal Volumen, umgepolt
    d["EPU"] = d["EPU_vol"]                             # EPU: nur Volumen (Typ-1-Indikator)
    d["ECO"] = -(d["ECO_tone"]) * d["ECO_vol"]
    d["INT"] = -(d["INT_tone"]) * d["INT_vol"]

    for spalte in treiber:                             # Finanzdaten fehlen am Wochenende, GDELT nicht
        d[spalte] = d[spalte].ffill()                  # Lücken mit dem Vortageswert füllen

    for spalte in treiber:                             # jeden Treiber glätten und standardisieren
        d[spalte + "_z"] = glaetten_standardisieren(d[spalte])

    d["cds_glatt"] = d["cds"].rolling(28, min_periods=1).mean()   # auch den CDS-Spread glätten

    Xspalten = [t + "_z" for t in treiber]             # die standardisierten Treiber fürs Training
    d = d.dropna(subset=Xspalten + ["cds_glatt"]).reset_index(drop=True)   # Zeilen mit Lücken entfernen

    train = d[d["Date"] < KRIEG - pd.Timedelta(days=28)]    # nur Daten bis 28 Tage vor dem Krieg (Puffer gegen Leakage)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)   # Random Forest, fester Zufall für gleiche Ergebnisse
    rf.fit(train[Xspalten], train["cds_glatt"])        # Modell trainieren

    mae = mean_absolute_error(train["cds_glatt"], rf.predict(train[Xspalten]))   # Trainingsfehler messen
    print(f"{land}: n_train={len(train)}, MAE={mae:.3f}")
    ergebnisse[land] = (d, rf, Xspalten)               # Ergebnis speichern



beschriftung = {"GPR_z": "GPR (Geopolitik)", "EPU_z": "EPU (Unsicherheit)", "ECO_z": "ECO (Wirtschaft)",
                "INT_z": "INT (Zinsen)", "VIX_z": "VIX (Volatilität)", "UST2Y_z": "US-Zins (2J)"}   # Legendentexte
farben = {"GPR_z": "#c0392b", "EPU_z": "#27ae60", "ECO_z": "#f39c12",
          "INT_z": "#8e44ad", "VIX_z": "#2980b9", "UST2Y_z": "#34495e"}   # Farben je Treiber

fig, axes = plt.subplots(1, len(ergebnisse), figsize=(7 * len(ergebnisse), 5))   # eine Spalte je Land
if len(ergebnisse) == 1:
    axes = [axes]

for ax, (land, (d, rf, Xspalten)) in zip(axes, ergebnisse.items()):
    fenster = d[(d["Date"] >= W0) & (d["Date"] <= W1)].copy()   # nur das Zeitfenster der Grafik
    sv = shap.TreeExplainer(rf).shap_values(fenster[Xspalten])   # Shapley-Werte berechnen
    sv = pd.DataFrame(sv, columns=Xspalten, index=fenster["Date"]).rolling(7, min_periods=1).mean()   # über 7 Tage glätten
    unten_pos = np.zeros(len(sv))                      # Startlinie für positive Flächen
    unten_neg = np.zeros(len(sv))                      # Startlinie für negative Flächen
    for spalte in Xspalten:
        werte = sv[spalte].values
        pos = np.clip(werte, 0, None)                  # nur die positiven Beiträge
        neg = np.clip(werte, None, 0)                  # nur die negativen Beiträge
        ax.fill_between(sv.index, unten_pos, unten_pos + pos, color=farben[spalte], label=beschriftung[spalte])   # positive nach oben stapeln
        ax.fill_between(sv.index, unten_neg, unten_neg + neg, color=farben[spalte])   # negative nach unten stapeln
        unten_pos += pos
        unten_neg += neg
    ax.axvline(KRIEG, color="black", linestyle="--", linewidth=1)   # Linie am Kriegsausbruch
    ax.text(KRIEG, ax.get_ylim()[1] * 0.95, " Kriegsausbruch", rotation=90, va="top", ha="right", fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator())   # ein Häkchen pro Monat
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))   # Datumsformat 
    ax.set_title(land, fontweight="bold")
    ax.set_ylabel("7-Tage-Durchschnitt der Shapley-Beiträge")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=7)

fig.suptitle("Treiber der CDS-Spreads im Israel-Iran-Krieg 2026", y=1.02)
plt.tight_layout()
plt.savefig("shapley_6treiber.png", dpi=150, bbox_inches="tight")   # Grafik speichern
print("Grafik gespeichert als shapley_6treiber.png")