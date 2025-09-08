using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[ExecuteAlways]
public class WarehouseZonesVisualizer : MonoBehaviour
{
    [Header("Grid settings")]
    public int gridSize = 12;
    public float cellSize = 1f;            // 1 unit = 1 cell
    public float markerHeight = 0.5f;      // visual Y (matches server y=0.5)

    [Header("Zones (grid coords)")]
    public Vector2Int dropZone = new Vector2Int(11, 11);
    public Vector2Int waitZone = new Vector2Int(0, 0);
    public Vector2Int loadZone = new Vector2Int(0, 11);
    public int zoneRadius = 1;

    [Header("Gizmos and runtime markers")]
    public bool showGizmos = true;             // Scene view gizmos
    public bool showRuntimeMarkers = true;     // Spawn primitive markers in Play mode
    public bool drawWire = true;
    public float gizmoAlpha = 0.25f;

    // internal marker objects (runtime)
    List<GameObject> _markers = new List<GameObject>();

    void OnValidate()
    {
        // Keep things sane in editor
        gridSize = Mathf.Max(1, gridSize);
        cellSize = Mathf.Max(0.01f, cellSize);
        markerHeight = Mathf.Max(0.01f, markerHeight);
        zoneRadius = Mathf.Max(0, zoneRadius);

        // update markers in editor or next Play
        if (!Application.isPlaying)
        {
            CleanupMarkers();
        }
    }

    void Start()
    {
        if (Application.isPlaying && showRuntimeMarkers)
            CreateOrUpdateRuntimeMarkers();
    }

    void OnDestroy()
    {
        CleanupMarkers();
    }

    void Update()
    {
        // in Play, keep markers updated (if you change zones at runtime)
        if (Application.isPlaying && showRuntimeMarkers)
        {
            // cheap: recreate when needed
            if (_markers.Count == 0) CreateOrUpdateRuntimeMarkers();
        }
    }

    // Convert grid cell (x,z) -> world position (x, y, z)
    Vector3 GridToWorld(Vector2Int gridCell)
    {
        return new Vector3(gridCell.x * cellSize, markerHeight, gridCell.y * cellSize);
    }

    // Draw gizmos (editor and optionally runtime Scene view)
    void OnDrawGizmos()
    {
        if (!showGizmos) return;

        // drop zone: red
        Color dropColor = new Color(1f, 0.3f, 0.3f, gizmoAlpha);
        DrawZone(dropZone, dropColor, "DROP");

        // wait zone: green
        Color waitColor = new Color(0.3f, 1f, 0.3f, gizmoAlpha);
        DrawZone(waitZone, waitColor, "WAIT");

        // load zone: blue
        Color loadColor = new Color(0.3f, 0.6f, 1f, gizmoAlpha);
        DrawZone(loadZone, loadColor, "LOAD");

        // optional: draw grid outline
        Gizmos.color = Color.gray;
        var gridWorldSize = new Vector3((gridSize - 1) * cellSize, 0f, (gridSize - 1) * cellSize);
        Vector3 origin = transform.position + new Vector3(0, 0.01f, 0);
        Gizmos.DrawWireCube(origin + gridWorldSize * 0.5f, gridWorldSize + new Vector3(cellSize, 0f, cellSize));
    }

    void DrawZone(Vector2Int zone, Color color, string label)
    {
        Vector3 center = transform.position + GridToWorld(zone);
        if (drawWire)
        {
            Gizmos.color = Color.Lerp(color, Color.black, 0.2f);
            Gizmos.DrawWireCube(center, new Vector3((zoneRadius * 2 + 1) * cellSize, 0.01f, (zoneRadius * 2 + 1) * cellSize));
        }

        // filled
        Gizmos.color = color;
        Gizmos.DrawCube(center, new Vector3((zoneRadius * 2 + 1) * cellSize, 0.01f, (zoneRadius * 2 + 1) * cellSize));

        // label in editor (safe)
#if UNITY_EDITOR
        Handles.Label(center + Vector3.up * 0.3f, label + $" ({zone.x},{zone.y})");
#endif
    }

    // Create primitive markers as children so they are visible in Game view during Play
    void CreateOrUpdateRuntimeMarkers()
    {
        CleanupMarkers();

        // small helper to create marker
        void CreateMarker(Vector2Int zone, Color color, string name)
        {
            Vector3 pos = transform.position + GridToWorld(zone);
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = $"__zone_marker_{name}";
            go.transform.position = pos;
            float s = Mathf.Max(0.2f * cellSize, cellSize * 0.9f);
            go.transform.localScale = new Vector3((zoneRadius * 2 + 1) * cellSize, 0.2f, (zoneRadius * 2 + 1) * cellSize);
            go.transform.SetParent(transform, true);
            // color the material (creates an instance)
            var rend = go.GetComponent<Renderer>();
            if (rend != null)
            {
                rend.material = new Material(Shader.Find("Standard"));
                rend.material.color = color;
                rend.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            }
            // disable collider to avoid physics interactions
            var col = go.GetComponent<Collider>();
            if (col != null) Destroy(col);

            _markers.Add(go);
        }

        CreateMarker(dropZone, new Color(1f, 0.3f, 0.3f, 0.6f), "DROP");
        CreateMarker(waitZone, new Color(0.3f, 1f, 0.3f, 0.6f), "WAIT");
        CreateMarker(loadZone, new Color(0.3f, 0.6f, 1f, 0.6f), "LOAD");
    }

    void CleanupMarkers()
    {
        for (int i = _markers.Count - 1; i >= 0; i--)
        {
            if (_markers[i] != null)
            {
#if UNITY_EDITOR
                // in editor DestroyImmediate is fine
                if (!Application.isPlaying) DestroyImmediate(_markers[i]);
                else Destroy(_markers[i]);
#else
                Destroy(_markers[i]);
#endif
            }
        }
        _markers.Clear();
    }
}
