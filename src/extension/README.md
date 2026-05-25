# src/extension/

Foxglove Studio extension that consumes `(device_id, ts_ns)` results from the query layer and jumps the active MCAP playback head to the exact moment.

> **Status:** scaffold / stretch goal. The deeplink URL scheme (`foxglove://open?...`) already works end-to-end from the search server — this extension is for the in-Studio panel workflow.

## Planned structure

```
src/extension/
├── package.json
├── tsconfig.json
└── src/
    ├── index.ts         (panel registration)
    ├── QueryPanel.tsx   (query box + result list)
    └── api.ts           (calls hybrid_cli over a local HTTP shim)
```

Built with `@foxglove/extension` against the Studio public API. See the Foxglove Studio extension SDK docs for current scaffolding commands.
