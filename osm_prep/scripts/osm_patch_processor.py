from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import math
import time
import warnings
import inspect
import hashlib
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize


def make_disk(radius_pixels: int) -> np.ndarray:
    radius_pixels = int(radius_pixels)
    if radius_pixels < 0:
        raise ValueError('Road radius must be nonnegative')
    grid = np.arange(-radius_pixels, radius_pixels + 1)
    yy, xx = np.meshgrid(grid, grid)
    return (xx * xx + yy * yy) <= (radius_pixels * radius_pixels)


def thicken_mask(mask: np.ndarray, radius_pixels: int = 3) -> np.ndarray:
    structure = make_disk(radius_pixels)
    mask_bool = mask > 0
    padded = np.pad(mask_bool, radius_pixels, mode='constant', constant_values=False)
    thickened = np.zeros_like(mask_bool, dtype=bool)
    for dy, dx in np.argwhere(structure):
        dy -= radius_pixels
        dx -= radius_pixels
        thickened |= padded[
            radius_pixels + dy:radius_pixels + dy + mask_bool.shape[0],
            radius_pixels + dx:radius_pixels + dx + mask_bool.shape[1],
        ]
    return thickened.astype(np.uint8)


class OSMPatchProcessor:
    """
    Download, explore, filter, visualise, export, and rasterise OpenStreetMap
    features for image patches described in a metadata CSV.
    """

    DEFAULT_SEMANTIC_KEYS = [
        "building",
        "highway",
        "landuse",
        "natural",
        "waterway",
        "railway",
        "amenity",
        "barrier",
        "man_made",
    ]
    # Value predicates from extract-osm-masks-kaggle v22.
    CLASS_TAGS = {
        'highway': {'highway': True},
        'building': {'building': True},
        'water': {'natural': ['water', 'waterway'], 'landuse': ['reservoir', 'basin']},
        'trees': {'natural': ['wood', 'scrub', 'tree_row'], 'landuse': ['forest']},
    }

    REQUIRED_COLUMNS = [
        "filename", "crs", "width_px", "height_px",
        "left", "bottom", "right", "top",
        "left_wgs84", "bottom_wgs84", "right_wgs84", "top_wgs84",
    ]

    NON_TAG_COLUMNS = {
        "geometry", "osmid", "element", "id", "nodes",
        "version", "timestamp", "changeset", "user", "uid",
        "source_patch"
    }

    def __init__(
        self,
        patch_metadata: Union[str, Path, pd.DataFrame],
        output_dir: Union[str, Path] = "osm_patch_outputs",
        semantic_keys: Optional[Sequence[str]] = None,
        overpass_pause_seconds: float = 1.0,
        image_root: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.semantic_keys = (
            list(semantic_keys) if semantic_keys is not None else list(self.DEFAULT_SEMANTIC_KEYS)
        )
        self.overpass_pause_seconds = float(overpass_pause_seconds)
        self.image_root = Path(image_root) if image_root is not None else None
        self.verbose = bool(verbose)

        self.patch_df = self._load_patch_metadata(patch_metadata)
        self._validate_patch_metadata()

    def resolve_patch_image_path(self, patch_id: Union[int, str]) -> Path:
        row = self.get_patch_row(patch_id)
        filename = str(row["filename"])

        if "filepath" in row.index and pd.notna(row["filepath"]):
            csv_path = Path(str(row["filepath"]))
            if csv_path.is_file():
                return csv_path

        if self.image_root is not None:
            split_name = Path(filename).stem.rsplit('_', 1)[0]
            candidate_paths = [
                self.image_root / split_name / "images" / filename,
                self.image_root / split_name / filename,
                self.image_root / filename,
            ]
            for candidate in candidate_paths:
                if candidate.is_file():
                    return candidate

        raise FileNotFoundError(
            f"Patch image not found for '{filename}'.\n"
            f"Checked the CSV 'filepath' value"
            + (" and local image_root candidates." if self.image_root is not None else ".")
            + "\nTo fix this, either:\n"
            + "1. update the 'filepath' column in the patch metadata CSV, or\n"
            + "2. pass image_root='.../OpenEarthMap_wo_xBD' when creating OSMPatchProcessor."
        )

    def _load_patch_metadata(self, patch_metadata: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(patch_metadata, pd.DataFrame):
            df = patch_metadata.copy()
        else:
            df = pd.read_csv(patch_metadata)

        if "filename" in df.columns:
            df["filename"] = df["filename"].astype(str)
        return df

    def _validate_patch_metadata(self) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.patch_df.columns]
        if missing:
            raise ValueError(f"Patch metadata is missing required columns: {missing}")

    def get_patch_row(self, patch_id: Union[int, str]) -> pd.Series:
        if isinstance(patch_id, int):
            return self.patch_df.iloc[patch_id]

        sub = self.patch_df[self.patch_df["filename"] == str(patch_id)]
        if sub.empty:
            raise KeyError(f"Patch not found: {patch_id}")
        return sub.iloc[0]

    def get_patch_bounds_wgs84(self, patch_id: Union[int, str]) -> Dict[str, float]:
        row = self.get_patch_row(patch_id)
        bounds = {
            "west": float(row["left_wgs84"]),
            "south": float(row["bottom_wgs84"]),
            "east": float(row["right_wgs84"]),
            "north": float(row["top_wgs84"]),
        }
        if not (-180 <= bounds['west'] < bounds['east'] <= 180
                and -90 <= bounds['south'] < bounds['north'] <= 90):
            raise ValueError(f'Invalid WGS84 bounds for {patch_id}: {bounds}')
        return bounds

    def get_patch_profile(self, patch_id: Union[int, str], dtype: str = "uint8", count: int = 1) -> Dict:
        row = self.get_patch_row(patch_id)
        crs = rasterio.crs.CRS.from_user_input(row['crs'])
        width, height = int(row['width_px']), int(row['height_px'])
        if width <= 0 or height <= 0 or width != float(row['width_px']) or height != float(row['height_px']):
            raise ValueError(f'Invalid grid dimensions for {patch_id}')
        try:
            image_path = self.resolve_patch_image_path(patch_id)
        except FileNotFoundError:
            image_path = None
        if image_path is not None:
            with rasterio.open(image_path) as src:
                if src.crs != crs or src.width != width or src.height != height:
                    raise ValueError(f'Source grid disagrees with metadata for {patch_id}')
                transform = src.transform
        else:
            columns = [f'transform_{key}' for key in 'abcdef']
            if not all(key in row.index and pd.notna(row[key]) for key in columns):
                raise ValueError(f'Exact affine unavailable for {patch_id}; provide source TIFFs or re-extract metadata')
            transform = Affine(*(float(row[key]) for key in columns))
        if (not np.isfinite(tuple(transform)).all()
                or transform.a * transform.e - transform.b * transform.d == 0):
            raise ValueError(f'Unsupported affine grid for {patch_id}')
        corners = [transform * point for point in ((0, 0), (width, 0), (0, height), (width, height))]
        xs, ys = zip(*corners)
        bounds = (min(xs), min(ys), max(xs), max(ys))
        if not np.allclose(bounds, [row[k] for k in ('left', 'bottom', 'right', 'top')], rtol=0, atol=1e-6):
            raise ValueError(f'Affine bounds disagree with metadata for {patch_id}')
        return {
            "driver": "GTiff",
            "height": int(row["height_px"]),
            "width": int(row["width_px"]),
            "count": int(count),
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }

    def build_tags_filter(self, semantic_keys: Optional[Sequence[str]] = None) -> Dict:
        keys = list(semantic_keys) if semantic_keys is not None else self.semantic_keys
        tags = {}
        for name in keys:
            for key, values in self.CLASS_TAGS.get(name, {name: True}).items():
                if tags.get(key) is True or values is True:
                    tags[key] = True
                else:
                    tags[key] = sorted(set(tags.get(key, []) + list(values)))
        return tags

    @staticmethod
    def _import_osmnx():
        try:
            import osmnx as ox
        except ImportError as e:
            raise ImportError(
                "osmnx is required for OSM downloading. Install it with:\n"
                "pip install osmnx"
            ) from e
        return ox

    def _download_features_osmnx_compat(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        tags: Dict[str, bool],
    ) -> gpd.GeoDataFrame:
        ox = self._import_osmnx()

        parameters = inspect.signature(ox.features_from_bbox).parameters
        if all(key in parameters for key in ('north', 'south', 'east', 'west')):
            return ox.features_from_bbox(
                north=north,
                south=south,
                east=east,
                west=west,
                tags=tags,
            )
        if 'bbox' in parameters:
            return ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
        raise RuntimeError('Unsupported OSMnx bbox interface')

    @staticmethod
    def _safe_vector_export_frame(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        out = gdf.copy()
        for col in out.columns:
            if col == "geometry":
                continue
            if str(out[col].dtype) != "object":
                continue

            def _normalise(value):
                if isinstance(value, (list, dict, tuple, set)):
                    return str(value)
                return value

            out[col] = out[col].map(_normalise)
        return out

    def download_patch_features(
        self,
        patch_id: Union[int, str],
        semantic_keys: Optional[Sequence[str]] = None,
        use_cache: bool = True,
        cache_format: str = "gpkg",
    ) -> gpd.GeoDataFrame:
        row = self.get_patch_row(patch_id)
        patch_name = Path(str(row["filename"])).stem
        self.get_patch_profile(patch_id)
        bounds = self.get_patch_bounds_wgs84(patch_id)
        tags = self.build_tags_filter(semantic_keys)
        query = {'bounds': bounds, 'tags': tags}
        query_id = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()[:16]

        cache_dir = self.output_dir / "cache_vectors"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_format = cache_format.lower()
        if cache_format == "geojson":
            cache_path = cache_dir / f"{patch_name}_{query_id}.geojson"
        else:
            cache_path = cache_dir / f"{patch_name}_{query_id}.gpkg"

        if use_cache and cache_path.exists():
            if self.verbose:
                print(f"Loading cached OSM features: {cache_path}")
            return gpd.read_file(cache_path)

        if self.verbose:
            print(f"Downloading OSM features for patch: {row['filename']}")
            print(f"WGS84 bounds: {bounds}")
            print(f"Semantic keys: {list(tags.keys())}")

        try:
            gdf = self._download_features_osmnx_compat(
                north=bounds['north'], south=bounds['south'],
                east=bounds['east'], west=bounds['west'], tags=tags,
            )
        except Exception as exc:
            # Only the library's explicit no-features response is a valid empty query.
            if (type(exc).__name__ == 'InsufficientResponseError'
                    and ('No data elements in server response' in str(exc) or 'No matching features' in str(exc))):
                gdf = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
            else:
                raise

        gdf = gdf.reset_index(drop=False)
        gdf = gdf.to_crs(row["crs"])
        gdf["source_patch"] = str(row["filename"])

        if use_cache and not gdf.empty:
            self.export_vector(gdf, cache_path)

        time.sleep(self.overpass_pause_seconds)
        return gdf

    def process_all_patches(
        self,
        patch_ids: Optional[Sequence[Union[int, str]]] = None,
        semantic_keys: Optional[Sequence[str]] = None,
        use_cache: bool = True,
        cache_format: str = "gpkg",
        save_key_summary: bool = True,
        save_top_values: bool = True,
        top_n: int = 20,
        continue_on_error: bool = True,
        return_results: bool = False,
    ) -> Optional[Dict[str, Union[Dict, Exception]]]:
        if patch_ids is None:
            patch_ids = self.patch_df["filename"].tolist()

        summary_dir = self.output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for patch_id in patch_ids:
            try:
                row = self.get_patch_row(patch_id)
                patch_name = Path(str(row["filename"])).stem

                if self.verbose:
                    print(f"\nProcessing patch: {row['filename']}")

                gdf = self.download_patch_features(
                    patch_id=patch_id,
                    semantic_keys=semantic_keys,
                    use_cache=use_cache,
                    cache_format=cache_format,
                )

                key_summary_df = self.summarise_tag_keys(gdf)

                selected_keys = semantic_keys if semantic_keys is not None else self.semantic_keys
                top_values_dict = self.get_top_values_for_selected_keys(
                    gdf=gdf,
                    selected_keys=selected_keys,
                    top_n=top_n,
                )

                if save_key_summary:
                    key_summary_path = summary_dir / f"{patch_name}_key_summary.csv"
                    key_summary_df.to_csv(key_summary_path, index=False)

                if save_top_values:
                    top_rows = []
                    for key, df_ in top_values_dict.items():
                        tmp = df_.copy()
                        tmp["patch"] = row["filename"]
                        top_rows.append(tmp)

                    if top_rows:
                        top_values_df = pd.concat(top_rows, ignore_index=True)
                        top_values_path = summary_dir / f"{patch_name}_top_values.csv"
                        top_values_df.to_csv(top_values_path, index=False)

                if return_results:
                    results[row["filename"]] = {
                        "gdf": gdf,
                        "key_summary": key_summary_df,
                        "top_values": top_values_dict,
                    }

                if self.verbose:
                    print(f"Done: {row['filename']} | features: {len(gdf):,}")

            except Exception as e:
                if continue_on_error:
                    warnings.warn(f"Failed for patch {patch_id}: {e}")
                    if return_results:
                        results[str(patch_id)] = e
                else:
                    raise

        if return_results:
            return results

        return None

    def summarise_tag_keys(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        rows = []
        for col in gdf.columns:
            if col in self.NON_TAG_COLUMNS:
                continue
            count = int(gdf[col].notna().sum())
            if count > 0:
                rows.append({"key": col, "count": count})

        if not rows:
            return pd.DataFrame(columns=["key", "count"])

        return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

    def get_top_values_for_selected_keys(
        self,
        gdf: gpd.GeoDataFrame,
        selected_keys: Optional[Sequence[str]] = None,
        top_n: int = 20,
    ) -> Dict[str, pd.DataFrame]:
        keys = list(selected_keys) if selected_keys is not None else self.semantic_keys
        results = {}

        for key in keys:
            if key not in gdf.columns:
                continue

            sub = gdf[gdf[key].notna()][key].astype(str).value_counts().head(int(top_n))
            if len(sub) == 0:
                continue

            results[key] = (
                sub.rename_axis("value")
                .reset_index(name="count")
                .assign(key=key)[["key", "value", "count"]]
            )

        return results

    def _normalise_rgb_for_display(
        self,
        rgb: np.ndarray,
        clip_percentiles: tuple = (2, 98),
    ) -> np.ndarray:
        rgb = rgb.astype(np.float32)

        if rgb.max() > 1.0:
            low = np.percentile(rgb, clip_percentiles[0], axis=(0, 1), keepdims=True)
            high = np.percentile(rgb, clip_percentiles[1], axis=(0, 1), keepdims=True)

            denom = np.where((high - low) == 0, 1.0, (high - low))
            rgb = (rgb - low) / denom
            rgb = np.clip(rgb, 0, 1)

        return rgb

    def read_patch_rgb(
        self,
        patch_id: Union[int, str],
        bands: Sequence[int] = (1, 2, 3),
        normalise: bool = True,
        clip_percentiles: tuple = (2, 98),
    ) -> np.ndarray:
        image_path = self.resolve_patch_image_path(patch_id)

        with rasterio.open(image_path) as src:
            arr = src.read(list(bands))

        rgb = np.moveaxis(arr, 0, -1)

        if normalise:
            rgb = self._normalise_rgb_for_display(
                rgb=rgb,
                clip_percentiles=clip_percentiles,
            )

        return rgb

    @staticmethod
    def _style_subplot_frame(ax, lw: float = 0.6) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(lw)
            spine.set_edgecolor("black")

    def plot_patch_rgb_and_keys(
        self,
        patch_id: Union[int, str],
        patch_gdf: Optional[gpd.GeoDataFrame] = None,
        keys: Optional[Sequence[str]] = None,
        bands: Sequence[int] = (1, 2, 3),
        normalise_rgb: bool = True,
        clip_percentiles: tuple = (2, 98),
        ncols: int = 3,
        panel_size: tuple = (5, 5),
        alpha: float = 0.85,
        linewidth: float = 0.4,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        row = self.get_patch_row(patch_id)
        patch_name = row["filename"]

        if keys is None:
            keys = self.semantic_keys

        rgb = None
        try:
            rgb = self.read_patch_rgb(
                patch_id=patch_id,
                bands=bands,
                normalise=normalise_rgb,
                clip_percentiles=clip_percentiles,
            )
        except FileNotFoundError as e:
            if self.verbose:
                print("\n[Warning] RGB image could not be loaded.")
                print(str(e))

        if patch_gdf is None:
            patch_gdf = self.download_patch_features(
                patch_id=patch_id,
                semantic_keys=keys,
                use_cache=True,
                cache_format="gpkg",
            )

        total_panels = 1 + len(keys)
        ncols = max(1, int(ncols))
        nrows = math.ceil(total_panels / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        )
        axes = np.array(axes).reshape(-1)

        if rgb is not None:
            axes[0].imshow(rgb)
            axes[0].set_title(f"RGB — {patch_name}", fontsize=11)
            self._style_subplot_frame(axes[0], lw=0.6)
        else:
            placeholder = np.ones((200, 200, 3), dtype=float)
            axes[0].imshow(placeholder)
            axes[0].set_title(f"RGB missing — {patch_name}", fontsize=11)
            axes[0].text(
                0.5,
                0.60,
                "RGB / TIFF image\nnot found",
                ha="center",
                va="center",
                fontsize=12,
                transform=axes[0].transAxes,
            )
            axes[0].text(
                0.5,
                0.28,
                "Update either:\n- CSV column: filepath\n- or class parameter: image_root",
                ha="center",
                va="center",
                fontsize=9,
                transform=axes[0].transAxes,
            )
            self._style_subplot_frame(axes[0], lw=0.8)

        default_colours = {
            "building": "lightgrey",
            "highway": "darkred",
            "landuse": "forestgreen",
            "natural": "seagreen",
            "waterway": "royalblue",
            "railway": "black",
            "amenity": "orange",
            "barrier": "purple",
            "man_made": "saddlebrown",
        }

        for i, key in enumerate(keys, start=1):
            ax = axes[i]
            sub_gdf = self.filter_features_by_key(gdf=patch_gdf, key=key)

            if sub_gdf.empty:
                ax.set_title(f"{key} — no features", fontsize=11)
                self._style_subplot_frame(ax, lw=0.8)
                continue

            colour = default_colours.get(key, "grey")
            sub_gdf.plot(
                ax=ax,
                color=colour,
                edgecolor=colour,
                linewidth=linewidth,
                alpha=alpha,
            )
            ax.set_title(f"{key} — {patch_name}", fontsize=11)
            self._style_subplot_frame(ax, lw=0.6)

        for j in range(total_panels, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()

    def build_presence_only_class_specs(
        self,
        selected_keys: Optional[Sequence[str]] = None,
        geometry_type_map: Optional[Dict[str, Sequence[str]]] = None,
        buffer_map: Optional[Dict[str, Optional[float]]] = None,
    ) -> Dict[str, Dict]:
        keys = list(selected_keys) if selected_keys is not None else list(self.semantic_keys)
        geometry_type_map = geometry_type_map or {}
        buffer_map = buffer_map or {}

        class_specs = {}
        for key in keys:
            class_specs[key] = {
                "key": key,
                "tags": self.CLASS_TAGS.get(key, {key: True}),
                "allowed_values": None,
                "geometry_types": geometry_type_map.get(key),
                "buffer_meters": buffer_map.get(key),
            }

        return class_specs

    def filter_features_by_key(
        self,
        gdf: gpd.GeoDataFrame,
        key: str,
        allowed_values: Optional[Sequence[str]] = None,
        geometry_types: Optional[Sequence[str]] = None,
    ) -> gpd.GeoDataFrame:
        if key not in gdf.columns:
            return gdf.iloc[0:0].copy()

        out = gdf[gdf[key].notna()].copy()

        if allowed_values is not None:
            allowed = set(map(str, allowed_values))
            out = out[out[key].astype(str).isin(allowed)].copy()

        if geometry_types is not None:
            allowed_geom = set(geometry_types)
            out = out[out.geometry.geom_type.isin(allowed_geom)].copy()

        return out

    @staticmethod
    def filter_features_by_tags(gdf, tags):
        selected = pd.Series(False, index=gdf.index)
        for key, values in tags.items():
            if key in gdf.columns:
                matches = gdf[key].notna() if values is True else gdf[key].isin(values)
                selected |= matches
        return gdf[selected].copy()

    @staticmethod
    def geometry_type_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        if gdf.empty:
            return pd.DataFrame(columns=["geometry_type", "count"])

        return (
            gdf.geometry.geom_type.value_counts()
            .rename_axis("geometry_type")
            .reset_index(name="count")
        )

    def plot_features(
        self,
        gdf: gpd.GeoDataFrame,
        title: str,
        figsize: tuple = (10, 10),
        color: str = "lightgrey",
        edgecolor: str = "black",
        linewidth: float = 0.3,
        alpha: float = 0.9,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if gdf.empty:
            print("No features to plot.")
            return

        fig, ax = plt.subplots(figsize=figsize)
        gdf.plot(
            ax=ax,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()

    def export_vector(self, gdf: gpd.GeoDataFrame, output_path: Union[str, Path]) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        safe_gdf = self._safe_vector_export_frame(gdf)
        suffix = output_path.suffix.lower()

        if suffix == ".gpkg":
            safe_gdf.to_file(output_path, driver="GPKG")
        elif suffix in [".geojson", ".json"]:
            safe_gdf.to_file(output_path, driver="GeoJSON")
        else:
            raise ValueError(f"Unsupported vector output extension: {suffix}")

    def _prepare_geometry_for_raster(
        self,
        gdf: gpd.GeoDataFrame,
        patch_id: Union[int, str],
        buffer_meters: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        row = self.get_patch_row(patch_id)
        out = gdf.copy()

        if out.empty:
            return out

        if str(out.crs) != str(row["crs"]):
            out = out.to_crs(row["crs"])

        if buffer_meters is not None and float(buffer_meters) > 0:
            out["geometry"] = out.geometry.buffer(float(buffer_meters))

        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
        return out

    def create_binary_masks(
        self,
        patch_id: Union[int, str],
        gdf: gpd.GeoDataFrame,
        class_specs: Dict[str, Dict],
        output_dir: Optional[Union[str, Path]] = None,
        all_touched: bool = False,
        save_png: bool = False,
        road_buffer_pixels: int = 3,
    ) -> Dict[str, np.ndarray]:
        row = self.get_patch_row(patch_id)
        profile = self.get_patch_profile(patch_id, dtype="uint8", count=1)
        transform = profile["transform"]
        height = profile["height"]
        width = profile["width"]

        if output_dir is None:
            output_dir = self.output_dir / "binary_masks" / Path(str(row["filename"])).stem
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        masks = {}

        for class_name, spec in class_specs.items():
            key = spec["key"]
            allowed_values = spec.get("allowed_values")
            geometry_types = spec.get("geometry_types")
            buffer_meters = spec.get("buffer_meters")

            if 'tags' in spec:
                sub = self.filter_features_by_tags(gdf, spec['tags'])
                if geometry_types is not None:
                    sub = sub[sub.geometry.geom_type.isin(geometry_types)].copy()
            else:
                sub = self.filter_features_by_key(gdf, key, allowed_values, geometry_types)
            sub = self._prepare_geometry_for_raster(
                gdf=sub,
                patch_id=patch_id,
                buffer_meters=buffer_meters,
            )

            if sub.empty:
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                shapes = ((geom, 1) for geom in sub.geometry)
                mask = rasterize(
                    shapes=shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    default_value=1,
                    dtype="uint8",
                    all_touched=all_touched,
                )

            if class_name in ('highway', 'roads'):
                mask = thicken_mask(mask, road_buffer_pixels)
            masks[class_name] = mask

            tif_path = output_dir / f"{Path(str(row['filename'])).stem}_{class_name}_mask.tif"
            if tif_path.exists():
                raise FileExistsError(f'Refusing to overwrite mask: {tif_path}')
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(mask, 1)

            if save_png:
                preview_dir = output_dir / 'previews'
                preview_dir.mkdir(exist_ok=True)
                png_path = preview_dir / f"{Path(str(row['filename'])).stem}_{class_name}_mask.png"
                plt.figure(figsize=(6, 6))
                plt.imshow(mask, cmap="gray")
                plt.title(f"{class_name} binary mask")
                plt.axis("off")
                plt.savefig(png_path, dpi=200, bbox_inches="tight")
                plt.close()

        return masks

    def create_multiclass_mask(
        self,
        patch_id: Union[int, str],
        gdf: gpd.GeoDataFrame,
        class_specs: Dict[str, Dict],
        class_id_map: Dict[str, int],
        priority_order: Sequence[str],
        output_path: Optional[Union[str, Path]] = None,
        all_touched: bool = False,
        save_png: bool = False,
    ) -> np.ndarray:
        row = self.get_patch_row(patch_id)
        profile = self.get_patch_profile(patch_id, dtype="uint8", count=1)
        transform = profile["transform"]
        height = profile["height"]
        width = profile["width"]

        mask = np.zeros((height, width), dtype=np.uint8)

        for class_name in priority_order:
            if class_name not in class_specs:
                continue
            if class_name not in class_id_map:
                raise KeyError(f"class_id_map is missing a value for class: {class_name}")

            spec = class_specs[class_name]
            allowed_values = spec.get("allowed_values")
            geometry_types = spec.get("geometry_types")
            buffer_meters = spec.get("buffer_meters")

            if 'tags' in spec:
                sub = self.filter_features_by_tags(gdf, spec['tags'])
                if geometry_types is not None:
                    sub = sub[sub.geometry.geom_type.isin(geometry_types)].copy()
            else:
                sub = self.filter_features_by_key(gdf, spec['key'], allowed_values, geometry_types)
            sub = self._prepare_geometry_for_raster(
                gdf=sub,
                patch_id=patch_id,
                buffer_meters=buffer_meters,
            )

            if sub.empty:
                continue

            shapes = ((geom, int(class_id_map[class_name])) for geom in sub.geometry)
            class_raster = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
                all_touched=all_touched,
            )

            mask[class_raster > 0] = class_raster[class_raster > 0]

        if output_path is None:
            patch_stem = Path(str(row["filename"])).stem
            output_path = (
                self.output_dir / "multiclass_masks" / patch_stem / f"{patch_stem}_multiclass_mask.tif"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)

        if save_png:
            png_path = output_path.with_suffix(".png")
            plt.figure(figsize=(6, 6))
            plt.imshow(mask)
            plt.title("Multiclass mask")
            plt.axis("off")
            plt.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close()

        return mask

    def export_binary_masks_for_patch(
        self,
        patch_id: Union[int, str],
        selected_keys: Optional[Sequence[str]] = None,
        patch_gdf: Optional[gpd.GeoDataFrame] = None,
        output_dir: Optional[Union[str, Path]] = None,
        geometry_type_map: Optional[Dict[str, Sequence[str]]] = None,
        buffer_map: Optional[Dict[str, Optional[float]]] = None,
        use_cache: bool = True,
        cache_format: str = "gpkg",
        all_touched: bool = False,
        save_png: bool = True,
        road_buffer_pixels: int = 3,
    ) -> Dict[str, np.ndarray]:
        keys = list(selected_keys) if selected_keys is not None else list(self.semantic_keys)

        if patch_gdf is None:
            patch_gdf = self.download_patch_features(
                patch_id=patch_id,
                semantic_keys=keys,
                use_cache=use_cache,
                cache_format=cache_format,
            )

        class_specs = self.build_presence_only_class_specs(
            selected_keys=keys,
            geometry_type_map=geometry_type_map,
            buffer_map=buffer_map,
        )

        return self.create_binary_masks(
            patch_id=patch_id,
            gdf=patch_gdf,
            class_specs=class_specs,
            output_dir=output_dir,
            all_touched=all_touched,
            save_png=save_png,
            road_buffer_pixels=road_buffer_pixels,
        )

    def export_multiclass_mask_for_patch(
        self,
        patch_id: Union[int, str],
        selected_keys: Optional[Sequence[str]] = None,
        patch_gdf: Optional[gpd.GeoDataFrame] = None,
        output_path: Optional[Union[str, Path]] = None,
        geometry_type_map: Optional[Dict[str, Sequence[str]]] = None,
        buffer_map: Optional[Dict[str, Optional[float]]] = None,
        class_id_map: Optional[Dict[str, int]] = None,
        priority_order: Optional[Sequence[str]] = None,
        use_cache: bool = True,
        cache_format: str = "gpkg",
        all_touched: bool = False,
        save_png: bool = True,
    ) -> np.ndarray:
        keys = list(selected_keys) if selected_keys is not None else list(self.semantic_keys)

        if patch_gdf is None:
            patch_gdf = self.download_patch_features(
                patch_id=patch_id,
                semantic_keys=keys,
                use_cache=use_cache,
                cache_format=cache_format,
            )

        class_specs = self.build_presence_only_class_specs(
            selected_keys=keys,
            geometry_type_map=geometry_type_map,
            buffer_map=buffer_map,
        )

        if class_id_map is None:
            class_id_map = {key: i + 1 for i, key in enumerate(keys)}

        if priority_order is None:
            priority_order = list(keys)

        if output_path is None:
            patch_stem = Path(str(self.get_patch_row(patch_id)["filename"])).stem
            output_path = (
                self.output_dir / "multiclass_masks" / patch_stem / f"{patch_stem}_multiclass_mask.tif"
            )

        return self.create_multiclass_mask(
            patch_id=patch_id,
            gdf=patch_gdf,
            class_specs=class_specs,
            class_id_map=class_id_map,
            priority_order=priority_order,
            output_path=output_path,
            all_touched=all_touched,
            save_png=save_png,
        )
