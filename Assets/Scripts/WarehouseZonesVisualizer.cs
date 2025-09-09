// WarehouseZonesVisualizer.cs
// visualizador de zonas para el warehouse
// - genera marcadores en scene y en game view para drop/wait/load
// - puede usar coordenadas world (x,y,z) o coordenadas de grid (gx,gz)
// - los marcadores creados son hijos del objeto y no tienen collider
// - editar propiedades en el inspector segun tu escena
// notas:
// - todos los comentarios estan en minuscula y sin acentos
// - si usas coordenadas en grid, ajusta origin y cellSize para mapear a world

using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[ExecuteAlways]
public class WarehouseZonesVisualizer : MonoBehaviour
{
    [Header("general")]
    [Tooltip("si true usa las posiciones world (x,y,z) definidas abajo; si false usa coordenadas de grid (gx,gz)")]
    public bool useWorldPositions = true;

    [Header("grid mapping (solo si useWorldPositions = false)")]
    [Tooltip("tamaño de celda en unidades world (1 unidad = 1 celda tipicamente)")]
    public float cellSize = 1f;
    [Tooltip("origen de la grilla: la celda (0,0) mapeada a este punto world")]
    public Vector2 origin = Vector2.zero; // origin.x => gx=0 -> world.x, origin.y => gz=0 -> world.z
    [Tooltip("altura visual para los marcadores (y)")]
    public float markerHeight = 0.5f;

    [Header("zones - usa world o grid segun useWorldPositions")]
    // si useWorldPositions = true usa los Vector3; si false usa los Vector2Int en coordenadas de grilla
    public Vector3 dropZoneWorld = new Vector3(-45f, 0.5f, -45f);
    public Vector3 waitZoneWorld = new Vector3(-45f, 0.5f, 45f);
    public Vector3 loadZoneWorld = new Vector3(-5f, 0.5f, 5f);

    [Space(6)]
    public Vector2Int dropZoneGrid = new Vector2Int(11, 11);
    public Vector2Int waitZoneGrid = new Vector2Int(0, 0);
    public Vector2Int loadZoneGrid = new Vector2Int(0, 11);

    [Header("visual")]
    [Tooltip("radio de la zona en celdas (si trabajas en grid). si usas world, el radio se interpreta en multiples de cellSize.")]
    public int zoneRadius = 1;
    public bool showGizmos = true;
    public bool showRuntimeMarkers = true;
    public bool drawWire = true;
    [Range(0f,1f)]
    public float gizmoAlpha = 0.25f;

    // controles internos
    List<GameObject> _markers = new List<GameObject>();

    void OnValidate()
    {
        cellSize = Mathf.Max(0.001f, cellSize);
        markerHeight = Mathf.Max(0.001f, markerHeight);
        zoneRadius = Mathf.Max(0, zoneRadius);

        // en editor limpiar para recrear si cambian parametros
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

    void Update()
    {
        if (Application.isPlaying && showRuntimeMarkers)
        {
            if (_markers.Count == 0) CreateOrUpdateRuntimeMarkers();
        }
    }

    // convierte una celda de grid (gx,gz) a world (x,y,z) usando origin y cellSize
    Vector3 GridToWorld(Vector2Int gridCell)
    {
        float x = (gridCell.x - origin.x) * cellSize;
        float z = (gridCell.y - origin.y) * cellSize;
        return new Vector3(x, markerHeight, z) + transform.position;
    }

    // si recibes world coords directas, ajustalas con transform.position
    Vector3 WorldPosToLocal(Vector3 world)
    {
        // asumimos world ya en coordenadas de escena; si quieres convertir desde otro espacio
        // ajusta aqui (por ejemplo si unity scene usa offset)
        return world + Vector3.up * markerHeight;
    }

    void OnDrawGizmos()
    {
        if (!showGizmos) return;

        // colores
        Color dropColor = new Color(1f, 0.3f, 0.3f, gizmoAlpha);
        Color waitColor = new Color(0.3f, 1f, 0.3f, gizmoAlpha);
        Color loadColor = new Color(0.3f, 0.6f, 1f, gizmoAlpha);

        if (useWorldPositions)
        {
            DrawZoneWorld(dropZoneWorld, dropColor, "DROP");
            DrawZoneWorld(waitZoneWorld, waitColor, "WAIT");
            DrawZoneWorld(loadZoneWorld, loadColor, "LOAD");
        }
        else
        {
            DrawZoneGrid(dropZoneGrid, dropColor, "DROP");
            DrawZoneGrid(waitZoneGrid, waitColor, "WAIT");
            DrawZoneGrid(loadZoneGrid, loadColor, "LOAD");
        }
    }

    void DrawZoneGrid(Vector2Int zone, Color color, string label)
    {
        Vector3 center = GridToWorld(zone);
        float size = (zoneRadius * 2 + 1) * cellSize;
        if (drawWire)
        {
            Gizmos.color = Color.Lerp(color, Color.black, 0.2f);
            Gizmos.DrawWireCube(center, new Vector3(size, 0.01f, size));
        }
        Gizmos.color = color;
        Gizmos.DrawCube(center, new Vector3(size, 0.01f, size));
#if UNITY_EDITOR
        Handles.Label(center + Vector3.up * 0.3f, label + $" ({zone.x},{zone.y})");
#endif
    }

    void DrawZoneWorld(Vector3 worldPos, Color color, string label)
    {
        Vector3 center = WorldPosToLocal(worldPos);
        float size = (zoneRadius * 2 + 1) * cellSize;
        if (drawWire)
        {
            Gizmos.color = Color.Lerp(color, Color.black, 0.2f);
            Gizmos.DrawWireCube(center, new Vector3(size, 0.01f, size));
        }
        Gizmos.color = color;
        Gizmos.DrawCube(center, new Vector3(size, 0.01f, size));
#if UNITY_EDITOR
        Handles.Label(center + Vector3.up * 0.3f, label + $" {center.x:F1},{center.z:F1}");
#endif
    }

    // crea marcadores en runtime visibles en game view
    void CreateOrUpdateRuntimeMarkers()
    {
        CleanupMarkers();

        void CreateMarker(Vector3 pos, Color color, string name)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = $"__zone_marker_{name}";
            go.transform.position = pos;
            float s = Mathf.Max(0.2f * cellSize, cellSize * 0.9f);
            go.transform.localScale = new Vector3((zoneRadius * 2 + 1) * cellSize, 0.2f, (zoneRadius * 2 + 1) * cellSize);
            go.transform.SetParent(transform, true);

            var rend = go.GetComponent<Renderer>();
            if (rend != null)
            {
                // crear material instanciado para no afectar otros objetos
                rend.material = new Material(Shader.Find("Standard"));
                // hacer traslucido: ajustar modo al standar alpha
                rend.material.SetFloat("_Mode", 3f);
                rend.material.EnableKeyword("_ALPHAPREMULTIPLY_ON");
                Color c = color;
                c.a = 0.6f;
                rend.material.color = c;
                rend.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            }

            // quitar collider para evitar fisica
            var col = go.GetComponent<Collider>();
            if (col != null) DestroyImmediate(col);

            _markers.Add(go);
        }

        if (useWorldPositions)
        {
            CreateMarker(WorldPosToLocal(dropZoneWorld), new Color(1f, 0.3f, 0.3f, 0.6f), "DROP");
            CreateMarker(WorldPosToLocal(waitZoneWorld), new Color(0.3f, 1f, 0.3f, 0.6f), "WAIT");
            CreateMarker(WorldPosToLocal(loadZoneWorld), new Color(0.3f, 0.6f, 1f, 0.6f), "LOAD");
        }
        else
        {
            CreateMarker(GridToWorld(dropZoneGrid), new Color(1f, 0.3f, 0.3f, 0.6f), "DROP");
            CreateMarker(GridToWorld(waitZoneGrid), new Color(0.3f, 1f, 0.3f, 0.6f), "WAIT");
            CreateMarker(GridToWorld(loadZoneGrid), new Color(0.3f, 0.6f, 1f, 0.6f), "LOAD");
        }
    }

    void CleanupMarkers()
    {
        for (int i = _markers.Count - 1; i >= 0; i--)
        {
            if (_markers[i] != null)
            {
#if UNITY_EDITOR
                if (!Application.isPlaying) DestroyImmediate(_markers[i]);
                else Destroy(_markers[i]);
#else
                Destroy(_markers[i]);
#endif
            }
        }
        _markers.Clear();
    }

    void OnDestroy()
    {
        CleanupMarkers();
    }
}
