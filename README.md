#CDS-Shapley-Analyse: Israel-Iran-Krieg 2026

Dieser Code bildet die Methodik von Ortiz und Rodrigo (2025) vereinfacht nach. Er lädt CDS-Spreads und nachrichtenbasierte Indikatoren (GDELT) für Israel und Saudi-Arabien, trainiert einen Random Forest und zerlegt die Treiber des CDS-Spreads mit Shapley-Werten. Erstellt im Rahmen einer Seminararbeit am Alfred-Weber-Institut, Universität Heidelberg.

Daten:
Der Code erwartet die Datendateien in einem Unterordner daten/. Diese sind aus rechtlichen Gründen nicht im Repository enthalten und müssen selbst bezogen werden:

CDS-Spreads: investing.com
Nachrichtenindikatoren (GPR, EPU, ECO, INT): GDELT DOC-2.0-API (Suchbegriffe siehe Seminararbeit, Tabelle 1)
VIX und zweijährige US-Rendite: FRED (Federal Reserve)
