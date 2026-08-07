import 'package:moneylens/core/constants/app_constants.dart';
import 'package:moneylens/core/constants/category_constants.dart';
import 'package:moneylens/database/database_helper.dart';

class MerchantRepository {
  MerchantRepository._();
  static final MerchantRepository instance = MerchantRepository._();

  final _db = DatabaseHelper.instance;
  final String _table = AppConstants.tableMerchants;

  final Map<String, String> _cache = {};
  bool _cacheLoaded = false;

  Future<String> getCategory(String merchant) async {
    await _ensureCacheLoaded();
    final key = _normalizeMerchant(merchant);

    final exactMatch = _cache[key];
    if (exactMatch != null) return exactMatch;

    for (final entry in _cache.entries) {
      if (key.contains(entry.key) || entry.key.contains(key)) {
        return entry.value;
      }
    }

    return CategoryConstants.others;
  }

  Future<Map<String, String>> getAllMappings() async {
    await _ensureCacheLoaded();
    return Map.unmodifiable(_cache);
  }

  Future<Map<String, dynamic>?> getMerchant(String merchant) async {
    final key = _normalizeMerchant(merchant);
    final rows = await _db.query(
      _table,
      where: 'merchant = ?',
      whereArgs: [key],
      limit: 1,
    );
    return rows.isEmpty ? null : rows.first;
  }

  Future<void> saveMerchant(String merchant, String category) async {
    await upsertMerchant(merchant, category);
  }

  Future<void> updateMerchant(String merchant, String newCategory) async {
    final key = _normalizeMerchant(merchant);
    await _db.update(
      _table,
      {'category': newCategory},
      where: 'merchant = ?',
      whereArgs: [key],
    );
    _cache[key] = newCategory;
  }

  Future<void> upsertMerchant(String merchant, String category) async {
    final key = _normalizeMerchant(merchant);
    final updated = await _db.update(
      _table,
      {'category': category},
      where: 'merchant = ?',
      whereArgs: [key],
    );

    if (updated == 0) {
      await _db.insert(_table, {'merchant': key, 'category': category});
    }

    _cache[key] = category;
  }

  Future<void> invalidateCache() async {
    _cache.clear();
    _cacheLoaded = false;
    await _ensureCacheLoaded();
  }

  Future<void> _ensureCacheLoaded() async {
    if (_cacheLoaded) return;

    final rows = await _db.query(_table);
    for (final row in rows) {
      _cache[row['merchant'] as String] = row['category'] as String;
    }
    _cacheLoaded = true;
  }

  String _normalizeMerchant(String merchant) {
    return merchant.toLowerCase().trim();
  }
}
