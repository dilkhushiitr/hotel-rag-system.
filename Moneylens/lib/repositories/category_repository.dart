import 'package:moneylens/core/constants/app_constants.dart';
import 'package:moneylens/core/constants/category_constants.dart';
import 'package:moneylens/database/database_helper.dart';
import 'package:moneylens/models/transaction_model.dart';

class CategoryRepository {
  CategoryRepository._();
  static final CategoryRepository instance = CategoryRepository._();

  final _db = DatabaseHelper.instance;
  final String _table = AppConstants.tableTransactions;

  Future<List<CategorySummary>> getTotalsThisMonth() async {
    final now = DateTime.now();
    final start = DateTime(now.year, now.month, 1);
    final end = DateTime(now.year, now.month + 1, 1);
    return _getTotalsForRange(start, end);
  }

  Future<List<CategorySummary>> getTotalsForRange(
    DateTime from,
    DateTime to,
  ) async {
    return _getTotalsForRange(from, to);
  }

  Future<List<CategorySummary>> getAllTimeTotals() async {
    final rows = await _db.rawQuery('''
      SELECT
        category,
        SUM(amount) AS totalAmount,
        COUNT(*) AS transactionCount
      FROM $_table
      WHERE transactionType = 'debit'
      GROUP BY category
      ORDER BY totalAmount DESC
    ''');
    return rows.map(CategorySummary.fromMap).toList();
  }

  Future<List<String>> getUsedCategories() async {
    final rows = await _db.rawQuery('''
      SELECT DISTINCT category
      FROM $_table
      ORDER BY category ASC
    ''');
    return rows.map((row) => row['category'] as String).toList();
  }

  List<String> getAllCategories() => CategoryConstants.all;

  Future<String> getTopCategoryThisMonth() async {
    final totals = await getTotalsThisMonth();
    if (totals.isEmpty) return '—';
    return totals.first.category;
  }

  Future<List<TransactionModel>> getTransactionsByCategory(
    String category, {
    DateTime? from,
    DateTime? to,
  }) async {
    String where = 'category = ?';
    final args = <dynamic>[category];

    if (from != null) {
      where += ' AND date >= ?';
      args.add(from.millisecondsSinceEpoch);
    }

    if (to != null) {
      where += ' AND date < ?';
      args.add(to.millisecondsSinceEpoch);
    }

    final rows = await _db.query(
      _table,
      where: where,
      whereArgs: args,
      orderBy: 'date DESC',
    );
    return rows.map(TransactionModel.fromMap).toList();
  }

  Future<List<CategorySummary>> _getTotalsForRange(
    DateTime from,
    DateTime to,
  ) async {
    final rows = await _db.rawQuery(
      '''
      SELECT
        category,
        SUM(amount) AS totalAmount,
        COUNT(*) AS transactionCount
      FROM $_table
      WHERE transactionType = 'debit'
        AND date >= ?
        AND date < ?
      GROUP BY category
      ORDER BY totalAmount DESC
      ''',
      [from.millisecondsSinceEpoch, to.millisecondsSinceEpoch],
    );
    return rows.map(CategorySummary.fromMap).toList();
  }
}
