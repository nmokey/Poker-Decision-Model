CSS = """
        <style>
        .poker-table { overflow: hidden; margin: .5rem 0 1rem; border-radius: 1rem;
            border: 1px solid #dfe5ec; background: #f8fafc; box-shadow: 0 8px 24px #0f172a10; }
        .poker-table__header { display: flex; align-items: center; gap: .7rem;
            padding: .8rem 1rem; color: #475569; }
        .poker-street { border-radius: 999px; background: #e8eef8; color: #31466c;
            padding: .3rem .7rem; font-size: .75rem; font-weight: 800; letter-spacing: .08em; }
        .poker-position { margin-left: .25rem; font-size: .8rem; font-weight: 700; }
        .poker-felt { display: flex; flex-direction: column; align-items: center;
            gap: 1.1rem; padding: 1.35rem 1rem;
            background: radial-gradient(ellipse at center, #217a5a, #12563f 72%);
            color: white; }
        .poker-player, .poker-board { min-width: 0; text-align: center; }
        .poker-pot-display { display: flex; justify-content: center; min-height: 3.8rem; }
        .poker-board--center .poker-card-row, .poker-player--center .poker-card-row {
            justify-content: center; }
        .poker-player--center { display: flex; flex-direction: column; align-items: center; }
        .poker-chip-stat { display: inline-flex; align-items: center; gap: .55rem;
            border: 1px solid #ffffff38; border-radius: 999px; background: #092f255c;
            padding: .35rem .7rem .35rem .45rem; color: white; }
        .poker-chip-stack { position: relative; display: inline-block;
            width: 2rem; height: 1.5rem; }
        .poker-chip { position: absolute; left: .28rem; width: 1.25rem; height: .48rem;
            border: 2px dashed #ffffffd9; border-radius: 50%; box-shadow: 0 2px 2px #092f2570; }
        .poker-chip--blue { top: .05rem; background: #3978d4; }
        .poker-chip--red { top: .4rem; left: .12rem; background: #da4b50; }
        .poker-chip--gold { top: .75rem; background: #e3b341; }
        .poker-chip-stat__label { color: #d3eee2; font-size: .62rem;
            font-weight: 800; letter-spacing: .1em; }
        .poker-chip-stat__amount { font-size: .9rem; }
        .poker-label { display: block; margin-bottom: .55rem; color: #d3eee2;
            font-size: .68rem; font-weight: 800; letter-spacing: .12em; }
        .poker-card-row { display: flex; flex-wrap: wrap; gap: .45rem; }
        .poker-card {
            position: relative; display: flex; align-items: center; justify-content: center;
            width: 3.35rem; height: 4.65rem; border: 1px solid #d6d9df; border-radius: .55rem;
            background: linear-gradient(145deg, #fff, #f3f5f8);
            box-shadow: 0 2px 5px #0f172a20; font-family: Georgia, serif;
        }
        .poker-card__corner { position: absolute; top: .25rem; left: .35rem; font-size: .9rem;
            line-height: .95; font-weight: 700; }
        .poker-card__suit { font-size: 1.8rem; }
        .poker-card__corner--red, .poker-card__suit--red { color: #c6283d; }
        .poker-card__corner--black, .poker-card__suit--black { color: #202735; }
        .poker-card--back { border: 2px solid #aeb8c9; color: #77849a;
            background: repeating-linear-gradient(
                45deg, #e8edf5, #e8edf5 5px, #dce4ef 5px, #dce4ef 10px
            ); }
        .poker-action-history { grid-column: 1 / -1; width: fit-content; max-width: 100%;
            border: 1px solid #ffffff35; border-radius: 999px; background: #ffffff16;
            padding: .4rem .8rem; color: #f0f8f4; font-size: .82rem; }
        .poker-action-row { display: flex; flex-wrap: wrap; gap: .45rem; }
        .poker-action-chip { border: 1px solid #d5deea; border-radius: 999px;
            background: #f8fafc; padding: .38rem .75rem; color: #34445c;
            font-size: .82rem; font-weight: 650; }
        @media (max-width: 640px) {
            .poker-felt { gap: 1rem; padding: 1rem; }
            .poker-table__header { gap: .5rem; padding: .7rem; }
        }
        </style>
        """
